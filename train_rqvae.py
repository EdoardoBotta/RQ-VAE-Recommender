import gin
import os
import torch

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from data.utils import cycle
from data.utils import next_batch
from distributions.gumbel import TemperatureScheduler
from modules.rqvae import RqVae
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    max_grad_norm=1,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000,
    vae_input_dim=18,
    vae_embed_dim=32,
    vae_hidden_dim=32,
    vae_codebook_size=32,
    vae_n_layers=3
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    dataset = MovieLensMovieData(root=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = cycle(dataloader)
    dataloader = accelerator.prepare(dataloader)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=[vae_hidden_dim],
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = LinearLR(optimizer)

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    temp_scheduler = TemperatureScheduler(
        t0=1,
        min_t=0.1,
        anneal_rate=0.00003,
        step_size=6000
    )

    with tqdm(initial=0, total=iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            if iter == 0 and use_kmeans_init:
                init_data = next_batch(dataloader, device)
                model.kmeans_init(init_data)
            total_loss = 0
            t = temp_scheduler.get_t(iter)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(dataloader, device)

                with accelerator.autocast():
                    loss = model(data, gumbel_t=t)
                    loss = loss / gradient_accumulate_every
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            accelerator.wait_for_everyone()

            if (iter+1) % save_model_every == 0:
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
    train()
