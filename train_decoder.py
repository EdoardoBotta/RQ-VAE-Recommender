import gin

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from data.movie_lens import MovieLensSeqData
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
    movie_dataset_folder="dataset/ml-1m-movie",
    pretrained_rqvae_path=None,
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    vae_input_dim=18,
    vae_embed_dim=12,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_n_layers=3,
    attn_heads=16,
    attn_embed_dim=64,
    attn_layers=6
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    movie_dataset = MovieLensMovieData(root=movie_dataset_folder)
    dataset = MovieLensSeqData(root=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = cycle(dataloader)
    dataloader = accelerator.prepare(dataloader)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        rqvae_weights_path=pretrained_rqvae_path
    )
    tokenizer.precompute_corpus_ids(movie_dataset)

    import pdb; pdb.set_trace()

    model = DecoderRetrievalModel(
        embedding_dim=vae_embed_dim,
        d_out=vae_embed_dim,
        dropout=True,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=tokenizer.n_ids
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = LinearLR(optimizer)

    model, optimizer, tokenizer, scheduler = accelerator.prepare(
        model, optimizer, tokenizer, scheduler
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
                    loss = model(tokenized_data).loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            accelerator.wait_for_everyone()
            pbar.update(1)


if __name__ == "__main__":
    train(pretrained_rqvae_path="out/checkpoint_49999.pt")
