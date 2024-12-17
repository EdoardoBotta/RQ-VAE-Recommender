import gin
import os
import torch
import numpy as np
import wandb

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from data.movie_lens import MovieLensSize
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from distributions.gumbel import TemperatureScheduler
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from torch.optim import AdamW
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
    dataset_size=MovieLensSize._1M,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3
):
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    dataset = MovieLensMovieData(root=dataset_folder, dataset_size=dataset_size)
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
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if wandb_logging:
        wandb.login()
        run = wandb.init(
            project="rq-vae-training",
            config=params
        )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer.rq_vae = model

    temp_scheduler = TemperatureScheduler(
        t0=2,
        min_t=0.1,
        anneal_rate=0.00003,
        step_size=3000
    )

    with tqdm(initial=start_iter, total=start_iter+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], []]
        for iter in range(start_iter, start_iter+1+iterations):
            model.train()
            total_loss = 0
            t = 0.2 # temp_scheduler.get_t(iter)
            if iter == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(dataset[torch.arange(20000)], device)
                model(kmeans_init_data, t)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(dataloader, device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

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
            
            id_diversity_log = {}
            if (iter+1) % eval_every == 0 or iter+1 == iterations:
                tokenizer.reset()
                model.eval()

                corpus_ids = tokenizer.precompute_corpus_ids(dataset)
                max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                
                _, counts = torch.unique(corpus_ids[:,:-1], dim=0, return_counts=True)
                p = counts / corpus_ids.shape[0]
                rqvae_entropy = -(p*torch.log(p)).sum()

                id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()
            
            if wandb_logging:
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                }
                wandb.log({
                    "total_loss": total_loss.cpu().item(),
                    "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                    "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                    **emb_norms_avg_log,
                    **id_diversity_log
                })

            pbar.update(1)
    
    wandb.finish()


if __name__ == "__main__":
    train(
        iterations=50000,
        learning_rate=0.0001,
        weight_decay=0.01,
        batch_size=256,
        vae_input_dim=768,
        vae_n_cat_feats=0,
        vae_hidden_dims=[512, 256, 128],
        vae_embed_dim=64,
        vae_codebook_size=256,
        vae_codebook_normalize=False,
        vae_sim_vq=False,
        save_model_every=10000,
        eval_every=10000,
        dataset_folder="dataset/ml-32m",
        dataset_size=MovieLensSize._32M,
        save_dir_root="out/ml32m/",
        wandb_logging=True,
        commitment_weight=0.25,
        vae_n_layers=3,
        vae_codebook_mode=QuantizeForwardMode.ROTATION_TRICK,
    )
