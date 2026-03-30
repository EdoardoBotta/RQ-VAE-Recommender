"""
train_decoder_grid.py

Ablation study: trains GRID's HuggingFace T5 encoder-decoder on tokens produced
by the RQ-VAE tokenizer from RQ-VAE-Recommender. The goal is to isolate the
effect of the transformer architecture by holding the tokenizer fixed and
swapping the custom transformer from train_decoder.py for GRID's T5-based
SemanticIDEncoderDecoder.

Usage:
    python train_decoder_grid.py configs/decoder_grid_amazon.gin
"""

import os
import sys

# Add GRID repo to path so its `src` package is importable.
_GRID_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GRID")
)
sys.path.insert(0, _GRID_DIR)

import gin
import torch
import torch.nn as nn
import wandb

from typing import Optional
from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to, cycle, next_batch
from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics, parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDDecoderModule,
    SemanticIDEncoderDecoder as _SIED,
    SemanticIDEncoderModule,
    SemanticIDGenerativeRecommender as _SIDGR,
)


# ---------------------------------------------------------------------------
# Token format adapter
# ---------------------------------------------------------------------------

def _strip_dedup_col(tensor: torch.Tensor, sem_ids_dim: int, n_layers: int) -> torch.Tensor:
    """Strip the deduplication column appended by SemanticIdTokenizer.

    RQ-VAE tokenizer stores sem_ids with shape [B, N * (n_layers + 1)], where
    the last column per item is a corpus deduplication index, not a codebook
    token.  GRID's transformer expects exactly n_layers tokens per item.

    Args:
        tensor:      [B, N * sem_ids_dim]  where sem_ids_dim = n_layers + 1
        sem_ids_dim: tokens per item including the dedup column
        n_layers:    number of RQ-VAE codebook levels (= GRID's num_hierarchies)

    Returns:
        [B, N * n_layers]
    """
    B, total = tensor.shape
    N = total // sem_ids_dim
    return tensor.view(B, N, sem_ids_dim)[:, :, :n_layers].contiguous().view(B, N * n_layers)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRIDTransformer(nn.Module):
    """Standalone HuggingFace T5 encoder-decoder for sequential recommendation.

    Reuses GRID's SemanticIDEncoderDecoder architecture and its mathematical
    implementation without the PyTorch Lightning dependency, making it
    compatible with the RQ-VAE-Recommender accelerate-based training loop.

    Key architectural differences from EncoderDecoderRetrievalModel:
    - HuggingFace T5 encoder + T5 decoder (cross-attention) instead of custom
      TransformerEncoderDecoder blocks with Flash SDP and jagged tensors.
    - Per-hierarchy linear output heads instead of a single shared projection.
    - Separator token injected between items in the encoder input sequence.
    - Constrained beam search via prefix validity checking against the corpus.

    The mathematical methods are borrowed directly from GRID's classes at the
    class level so that any in-place logic (e.g. codebook device transfer) works
    correctly on GRIDTransformer instances without Lightning overhead.
    """

    # ---- Borrow mathematical implementations from GRID classes ----
    _add_repeating_offset_to_rows = _SIDGR._add_repeating_offset_to_rows
    _inject_sep_token_between_sids = _SIDGR._inject_sep_token_between_sids
    _check_valid_prefix = _SIDGR._check_valid_prefix
    _beam_search_one_step = _SIDGR._beam_search_one_step
    _is_kv_cache_valid = _SIDGR._is_kv_cache_valid
    encoder_forward_pass = _SIED.encoder_forward_pass
    decoder_forward_pass = _SIED.decoder_forward_pass
    generate = _SIED.generate
    get_embedding_table = _SIED.get_embedding_table
    forward = _SIED.forward

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        t5_d_model: int = 128,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        top_k_for_generation: int = 10,
        should_check_prefix: bool = True,
        should_add_sep_token: bool = True,
        num_user_bins: Optional[int] = None,
    ):
        """
        Args:
            codebooks: Corpus semantic IDs, shape [N_items, num_hierarchies].
                Built from SemanticIdTokenizer.cached_ids[:, :n_layers].
            num_hierarchies: Number of RQ-VAE codebook levels (n_layers).
            num_embeddings_per_hierarchy: Codebook size (e.g. 256).
            t5_d_model: T5 model hidden dimension.
            t5_num_heads: Number of T5 multi-head attention heads.
            t5_d_ff: T5 feed-forward inner dimension.
            t5_num_layers: Number of T5 encoder and decoder layers each.
            top_k_for_generation: Beam width for constrained beam search.
            should_check_prefix: Prune beams with invalid SID prefixes.
            should_add_sep_token: Inject separator token between items in
                the encoder input sequence (TIGER default: True).
            num_user_bins: If set, adds a hashed user embedding prepended to
                the encoder sequence.
        """
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = t5_d_model
        self.top_k_for_generation = top_k_for_generation
        self.should_check_prefix = should_check_prefix

        # Codebooks for constrained beam search: [N_items, num_hierarchies].
        # _check_valid_prefix may reassign this attribute to a different device.
        self.register_buffer("codebooks", codebooks)

        # ---- T5 Encoder ----
        encoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=False,
        )
        self.encoder = SemanticIDEncoderModule(encoder=T5EncoderModel(encoder_config))

        # ---- T5 Decoder ----
        decoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        # SemanticIDDecoderModule deletes embed_tokens right after construction;
        # newer transformers versions create it internally in T5Stack.__init__.
        t5_decoder = T5Stack(decoder_config)

        bos_token = nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
        decoder_mlp = nn.ModuleList(
            [
                nn.Linear(t5_d_model, num_embeddings_per_hierarchy, bias=False)
                for _ in range(num_hierarchies)
            ]
        )
        self.decoder = SemanticIDDecoderModule(
            decoder=t5_decoder,
            bos_token=bos_token,
            decoder_mlp=decoder_mlp,
        )

        # Shared encoder/decoder embedding table.
        # Tokens are offset by hierarchy level so all hierarchies share one table:
        # hierarchy h, token t  →  embedding index  h * codebook_size + t
        self.item_sid_embedding_table_encoder = nn.Embedding(
            num_embeddings=num_embeddings_per_hierarchy * num_hierarchies,
            embedding_dim=t5_d_model,
        )

        # Optional hashed user embedding (prepended to encoder sequence)
        self.user_embedding = (
            nn.Embedding(num_user_bins, t5_d_model) if num_user_bins else None
        )

        # Separator token injected after each item's tokens in encoder input
        self.sep_token = (
            nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
            if should_add_sep_token
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_loss(
    model: GRIDTransformer,
    tokenized_data,
    n_layers: int,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Teacher-forced per-hierarchy cross-entropy loss.

    Adapts the RQ-VAE TokenizedSeqBatch to the GRID model's input format by
    stripping the deduplication column and converting the boolean mask to long.

    The decoder is fed [BOS, h0, h1, …, h_{H-1}] and produces H+1 output
    positions.  After removing the last position (which has no target), output
    position h predicts hierarchy h of the next item.

    Args:
        model:           GRIDTransformer instance.
        tokenized_data:  TokenizedSeqBatch from SemanticIdTokenizer.
        n_layers:        Number of RQ-VAE codebook levels (num_hierarchies).
        loss_fn:         Cross-entropy loss (reduction='mean').

    Returns:
        Scalar loss tensor (sum over hierarchies, mean over batch).
    """
    sem_ids_dim = n_layers + 1  # RQ-VAE tokens per item (n_layers codebook + 1 dedup)

    # Strip dedup column: [B, N*(n_layers+1)] -> [B, N*n_layers]
    input_ids = _strip_dedup_col(tokenized_data.sem_ids, sem_ids_dim, n_layers)
    attention_mask = _strip_dedup_col(
        tokenized_data.seq_mask.long(), sem_ids_dim, n_layers
    )

    # Future item: strip dedup column -> [B, n_layers]
    fut_ids = tokenized_data.sem_ids_fut[:, :n_layers]

    # Teacher-forced forward pass.
    # Decoder input:  [BOS, h0_embed, h1_embed, …]  → shape [B, n_layers+1, d_model]
    # Decoder output: same shape before slicing
    decoder_output = model(
        attention_mask_encoder=attention_mask,
        input_ids=input_ids,
        future_ids=fut_ids,
    )

    # Remove the last output position so that position h predicts hierarchy h.
    # [B, n_layers+1, d_model] -> [B, n_layers, d_model]
    decoder_output = decoder_output[:, :-1]

    total_loss = torch.tensor(0.0, device=decoder_output.device)
    for h in range(n_layers):
        logits = model.decoder.decoder_mlp[h](decoder_output[:, h])  # [B, codebook_size]
        total_loss = total_loss + loss_fn(logits, fut_ids[:, h].long())

    return total_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    # ---- GRID T5 transformer hyperparameters ----
    t5_d_model=128,
    t5_num_heads=6,
    t5_d_ff=1024,
    t5_num_layers=4,
    top_k_for_generation=10,
    should_check_prefix=True,
    should_add_sep_token=True,
    num_user_bins=None,
    top_k_eval_list=[5, 10],
):
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="gen-retrieval-decoder-training-grid", config=params)

    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
    )
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=True,
        subsample=train_data_subsample,
        split=dataset_split,
    )
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split=dataset_split,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
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
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    # Build codebooks for constrained beam search.
    # tokenizer.cached_ids: [N_items, n_layers + 1]; strip the dedup column.
    codebooks = tokenizer.cached_ids[:, :vae_n_layers].cpu()

    loss_fn = nn.CrossEntropyLoss()

    model = GRIDTransformer(
        codebooks=codebooks,
        num_hierarchies=vae_n_layers,
        num_embeddings_per_hierarchy=vae_codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_check_prefix=should_check_prefix,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    model = torch.compile(model)

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    evaluator = SIDRetrievalEvaluator(
        metrics={"Recall": Recall, "NDCG": NDCG},
        top_k_list=top_k_eval_list,
    )
    evaluator.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    loss = _compute_loss(model, tokenized_data, vae_n_layers, loss_fn)
                    loss = loss / gradient_accumulate_every

                total_loss += loss.detach().item()

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

            assert model.item_sid_embedding_table_encoder.weight.grad is not None

            pbar.set_description(f"loss: {total_loss:.4f}")

            accelerator.wait_for_everyone()
            optimizer.step()
            lr_scheduler.step()
            accelerator.wait_for_everyone()

            # ---- Partial eval: loss only ----
            if (iter + 1) % partial_eval_every == 0:
                model.eval()
                eval_loss = 0.0
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)
                    with torch.no_grad():
                        eval_loss = _compute_loss(
                            model, tokenized_data, vae_n_layers, loss_fn
                        ).item()

                if wandb_logging and accelerator.is_main_process:
                    wandb.log({"eval_loss": eval_loss})

            # ---- Full eval: constrained beam search generation ----
            if (iter + 1) % full_eval_every == 0:
                model.eval()
                sem_ids_dim = vae_n_layers + 1

                with tqdm(
                    eval_dataloader,
                    desc=f"Eval {iter + 1}",
                    disable=not accelerator.is_main_process,
                ) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        input_ids = _strip_dedup_col(
                            tokenized_data.sem_ids, sem_ids_dim, vae_n_layers
                        )
                        attention_mask = _strip_dedup_col(
                            tokenized_data.seq_mask.long(), sem_ids_dim, vae_n_layers
                        )

                        with torch.no_grad():
                            generated_ids, marginal_probs = model.generate(
                                attention_mask=attention_mask,
                                input_ids=input_ids,
                            )
                        # generated_ids: [B, top_k, n_layers]
                        # actual: [B, n_layers] (strip dedup column from ground truth)
                        actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
                        evaluator(
                            marginal_probs=marginal_probs,
                            generated_ids=generated_ids,
                            labels=actual,
                        )

                eval_metrics = {
                    name: metric.compute().item()
                    for name, metric in evaluator.metrics.items()
                }
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                evaluator.reset()

            # ---- Checkpoint + wandb logging ----
            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)
                    torch.save(
                        {
                            "iter": iter,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                        },
                        save_dir_root + f"checkpoint_{iter}.pt",
                    )

                if wandb_logging:
                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss,
                            **train_debug_metrics,
                        }
                    )

            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
