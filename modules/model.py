import torch
import torch._dynamo

from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.tokenizer.semids import TokenizedSeqBatch
from modules.transformer.model import TransformerDecoder
from modules.utils import eval_mode
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
    def generate(self, batch: TokenizedSeqBatch, generate_length: int = 1, temperature: int = 1) -> GenerationOutput:
        B, N = batch.sem_ids.shape
        generated, log_probas = None, None
        batch_idx = torch.arange(B, device=batch.sem_ids.device)

        batch = TokenizedSeqBatch(*[v.detach().clone() for _, v in batch._asdict().items()])
        seq_mask = batch.seq_mask
        sem_ids = batch.sem_ids

        for _ in range(generate_length):
            next_token_pos = seq_mask.sum(axis=1)
            # Ensure at least self.sem_id_dim empty slots are available in every sequence
            # by rolling overflown sequences forward.
            to_shift = next_token_pos > N - self.sem_id_dim
            sem_ids[to_shift, :] = sem_ids[to_shift].roll(-self.sem_id_dim, dims=1)
            sem_ids[to_shift, N - self.sem_id_dim:] = -1
            seq_mask[to_shift, N - self.sem_id_dim:] = False

            next_token_pos = seq_mask.sum(axis=1)

            for _ in range(self.sem_id_dim):
                logits = self.forward(batch).logits
                probas = F.softmax(logits / temperature, dim=-1)
                samples = torch.multinomial(probas, num_samples=k)
                sampled_log_probas = torch.log(probas[batch_idx, samples.squeeze()]).unsqueeze(-1)

                generated = samples if generated is None else torch.cat([generated, samples], axis=-1)
                log_probas = (
                    sampled_log_probas if log_probas is None
                    else torch.cat([log_probas, sampled_log_probas], axis=-1)
                )
                sem_ids[batch_idx, next_token_pos] = samples.squeeze()
                seq_mask[batch_idx, next_token_pos] = True
                next_token_pos += 1
        
        return GenerationOutput(
            sem_ids=generated,
            log_probas=log_probas
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
