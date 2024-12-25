import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from data.movie_lens import MovieLensSeqData
from data.movie_lens import MovieLensSize
from data.utils import cycle
from data.utils import next_batch
from modules.model import DecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer
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
    dataset_size=MovieLensSize._1M,
    pretrained_rqvae_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4
):
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )

    movie_dataset = MovieLensMovieData(root=dataset_folder, dataset_size=dataset_size, force_process=force_dataset_process)
    dataset = MovieLensSeqData(root=dataset_folder, dataset_size=dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = cycle(dataloader)
    dataloader = accelerator.prepare(dataloader)

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
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(movie_dataset)

    # import pdb; pdb.set_trace()

    model = DecoderRetrievalModel(
        embedding_dim=vae_embed_dim,
        d_out=vae_embed_dim,
        dropout=False,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=dataset.max_seq_len*tokenizer.sem_ids_dim
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    with tqdm(initial=0, total=iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model.generate_next_sem_id(tokenized_data)
                    #loss = model(tokenized_data).loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

                accelerator.backward(total_loss)

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if wandb_logging:
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    train(
        iterations=20000,
        batch_size=64,
        vae_input_dim=768,
        vae_hidden_dims=[512, 256, 128],
        vae_embed_dim=64,
        vae_n_cat_feats=0,
        vae_codebook_size=256,
        wandb_logging=False,
        pretrained_rqvae_path="trained_models/checkpoint_high_entropy.pt",
        save_dir_root="out/decoder/",
        dataset_folder="dataset/ml-32m",
        dataset_size=MovieLensSize._32M,
        force_dataset_process=True
    )