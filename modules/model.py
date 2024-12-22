import torch
import torch._dynamo

from einops import rearrange
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.tokenizer.semids import TokenizedSeqBatch
from modules.transformer.model import TransformerDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import select_columns_per_row
from typing import NamedTuple
from torch import nn
from torch.nn import functional as F

# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision('high')


class ModelOutput(NamedTuple):
    loss: torch.Tensor
    logits: torch.Tensor


class GenerationOutput(NamedTuple):
    sem_ids: torch.Tensor
    log_probas: torch.Tensor


class DecoderRetrievalModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 d_out,
                 dropout,
                 num_heads,
                 n_layers,
                 num_embeddings,
                 sem_id_dim,
                 max_pos=2048) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        
        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)

        self.decoder = TransformerDecoder(
            d_in=embedding_dim,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=n_layers,
            do_cross_attn=False
        )

        self.out_proj = nn.Linear(d_out, num_embeddings, bias=False)
    
    def _predict(self, batch: TokenizedSeqBatch) -> torch.Tensor:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        
        B, N, D = sem_ids_emb.shape
        
        pos = torch.arange(N, device=sem_ids_emb.device)
        wpe = self.wpe(pos)

        input_embedding = user_emb.unsqueeze(1) + wpe.unsqueeze(0) + sem_ids_emb
        transformer_output = self.decoder(input_embedding)

        return transformer_output

    @torch.no_grad
    @eval_mode
    def generate_next_sem_id(self, batch: TokenizedSeqBatch, temperature: int = 1, top_k: bool = False) -> GenerationOutput:
        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 10 if top_k else 1
        n_top_k_candidates = 2*k if top_k else 1

        batch = TokenizedSeqBatch(*[v.detach().clone() for _, v in batch._asdict().items()])
        seq_mask = batch.seq_mask
        sem_ids = batch.sem_ids

        next_token_pos = seq_mask.sum(axis=1)
        # Ensure at least self.sem_id_dim empty slots are available in every sequence
        # by rolling overflown sequences forward.
        to_shift = next_token_pos > N - self.sem_id_dim
        sem_ids[to_shift, :] = sem_ids[to_shift].roll(-self.sem_id_dim, dims=1)
        sem_ids[to_shift, N - self.sem_id_dim:] = -1
        seq_mask[to_shift, N - self.sem_id_dim:] = False

        next_token_pos = seq_mask.sum(axis=1).repeat_interleave(k)

        for _ in range(self.sem_id_dim):
            logits = self.forward(batch).logits
            probas = F.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probas, num_samples=n_top_k_candidates)
            samples = samples.reshape(B, -1)
            probas = probas.reshape(B, -1)
            sampled_log_probas = torch.log(select_columns_per_row(probas, samples))

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                sampled_log_probas + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)
            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = select_columns_per_row(samples, top_k_indices)
            
            if generated is not None:
                parent_id = select_columns_per_row(generated, top_k_indices // n_top_k_candidates)
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                slice_length = generated.shape[2] + 1
                col_idx = (
                    rearrange(next_token_pos, "b -> b 1") -
                    torch.arange(slice_length-1, -1, -1, device=next_token_pos.device)
                )

                batch.sem_ids[
                    torch.arange(batch.sem_ids.shape[0], device=batch.sem_ids.device).unsqueeze(1),
                    col_idx
                ] = top_k_samples.flatten(end_dim=1)

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                sem_ids = batch.sem_ids.repeat_interleave(k, dim=0)
                sem_ids[
                    torch.arange(sem_ids.shape[0], device=sem_ids.device),
                    next_token_pos
                ] = top_k_samples.flatten()

                batch = TokenizedSeqBatch(
                    user_ids=batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=sem_ids,
                    seq_mask=seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())

            batch.seq_mask[
                torch.arange(batch.seq_mask.shape[0]),
                next_token_pos
            ] = True
            
            next_token_pos += 1

        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    # @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)

        if self.training:
            logits = self.out_proj(trnsf_out)
            target_mask = seq_mask[:, 1:]
            out = logits[:, :-1, :][target_mask, :]
            target = batch.sem_ids[:, 1:][target_mask]
            loss = F.cross_entropy(out, target)
        else:
            last_token_pos = seq_mask.sum(axis=-1) - 1
            logits = self.out_proj(trnsf_out[torch.arange(B, device=trnsf_out.device), last_token_pos])
            loss = None

        return ModelOutput(loss=loss, logits=logits)
