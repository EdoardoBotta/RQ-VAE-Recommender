import gin
import os
import torch
import numpy as np

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from data.utils import cycle
from data.utils import next_batch
from distributions.gumbel import TemperatureScheduler
from modules.rqvae import RqVae
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_n_layers=3
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    dataset = MovieLensMovieData(root=dataset_folder)
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None, collate_fn=lambda batch: batch)
    dataloader = cycle(dataloader)
    dataloader = accelerator.prepare(dataloader)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        n_layers=vae_n_layers,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    temp_scheduler = TemperatureScheduler(
        t0=2,
        min_t=0.1,
        anneal_rate=0.00003,
        step_size=3000
    )

    with tqdm(initial=0, total=iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = []
        for iter in range(start_iter+1, start_iter+1+iterations):
            model.train()
            total_loss = 0
            t = 0.2 # temp_scheduler.get_t(iter)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(dataloader, device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses.append(total_loss.cpu().item())
            losses = losses[-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses)

            pbar.set_description(f'loss: {print_loss:.4f}, t: {t:.3f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            accelerator.wait_for_everyone()

            if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                state = {
                    "iter": iter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }

                if not os.path.exists(save_dir_root):
                    os.makedirs(save_dir_root)

                torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")

            pbar.update(1)


if __name__ == "__main__":
    train(
        iterations=300000,
        learning_rate=0.0001,
        batch_size=256,
        vae_input_dim=786,
        vae_hidden_dims=[512, 256, 128],
        vae_embed_dim=32,
        vae_codebook_size=256,
        save_model_every=50000
    )
