import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import jagged_to_flattened_tensor
from modules.utils import maybe_repeat_interleave
from modules.utils import padded_to_jagged_tensor
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision('high')


@gin.constants_from_enum
class ModelType(Enum):
    DECODER = 1
    ENCODER_DECODER = 2


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class DecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.inference_verifier_fn = inference_verifier_fn
        
        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.decoder = TransformerDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=n_layers,
            do_cross_attn=False
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        
        B, N, D = sem_ids_emb.shape
          
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0) + self.decoder.seq_lengths
        wpe = self.wpe(pos)
        tte = self.tte(batch.token_type_ids)

        input_embedding = wpe + sem_ids_emb + tte

        if self.jagged_mode:
            seq_lengths = batch.seq_mask.sum(axis=1)
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths, max_len=input_embedding.shape[1])
        
        transformer_input = self.in_proj(input_embedding)
        transformer_output = self.decoder(transformer_input, padding_mask=batch.seq_mask, jagged=self.jagged_mode)

        return transformer_output

    @eval_mode
    @reset_kv_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 10 if top_k else 1
        n_top_k_candidates = 5*k if top_k else 1

        next_token_pos = batch.seq_mask.sum(axis=1)
        # Ensure at least self.sem_id_dim empty slots are available in every sequence
        # by rolling overflown sequences forward.
        to_shift = next_token_pos > N - self.sem_id_dim
        batch.sem_ids[to_shift, :] = batch.sem_ids[to_shift].roll(-self.sem_id_dim, dims=1)
        batch.sem_ids[to_shift, N - self.sem_id_dim:] = -1
        batch.seq_mask[to_shift, N - self.sem_id_dim:] = False

        next_token_pos = batch.seq_mask.sum(axis=1).repeat_interleave(k)

        for _ in range(self.sem_id_dim):
            logits = self.forward(batch).logits
            probas = F.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probas, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            samples = samples.reshape(B, -1)
            probas = probas.reshape(B, -1)
            sampled_log_probas = torch.log(select_columns_per_row(probas, samples))

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)
            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = select_columns_per_row(samples, top_k_indices)
            
            if generated is not None:
                parent_id = select_columns_per_row(generated, top_k_indices // n_top_k_candidates)
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                cache_idx = (
                    (torch.arange(B, device=top_k_indices.device).unsqueeze(-1)*k) +
                    top_k_indices // n_top_k_candidates
                ).flatten()

                self.decoder.apply_to_kv_cache(lambda x: x[cache_idx])

                batch = TokenizedSeqBatch(
                    user_ids=batch.user_ids,
                    sem_ids=top_k_samples[:, :, -1].reshape(-1, 1),
                    sem_ids_fut=batch.sem_ids_fut,
                    seq_mask=batch.seq_mask,
                    token_type_ids=batch.token_type_ids+1
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)
                next_batch_size = next_sem_ids.shape[0]

                batch = TokenizedSeqBatch(
                    user_ids=batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=next_sem_ids,
                    sem_ids_fut=batch.sem_ids_fut,
                    seq_mask=torch.ones(next_batch_size, 1, dtype=bool, device=next_sem_ids.device),
                    token_type_ids=torch.zeros(next_batch_size, 1, dtype=torch.int32, device=next_sem_ids.device)
                )

                self.decoder.apply_to_kv_cache(lambda x: x.repeat_interleave(k, axis=0))

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                logits = jagged_to_flattened_tensor(predict_out)
                target = torch.cat([
                    batch.sem_ids[:, 1:],
                    -torch.ones(B, 1, device=batch.sem_ids.device, dtype=batch.sem_ids.dtype)
                ], axis=1)[seq_mask]
                loss = F.cross_entropy(logits, target, ignore_index=-1)
            else:
                logits = predict_out
                target_mask = seq_mask[:, 1:]
                out = logits[:, :-1, :][target_mask, :]
                target = batch.sem_ids[:, 1:][target_mask]
                loss = F.cross_entropy(out, target)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            last_token_pos = trnsf_out.offsets()[1:]-1
            trnsf_out_flattened = jagged_to_flattened_tensor(trnsf_out)
            logits = self.out_proj(trnsf_out_flattened[last_token_pos, :])
            loss = None
        else:
            last_token_pos = seq_mask.sum(axis=-1) - 1
            logits = self.out_proj(trnsf_out[torch.arange(B, device=trnsf_out.device), last_token_pos])
            loss = None

        return ModelOutput(loss=loss, logits=logits)

class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,
            decoder_layers=n_layers // 2
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)
          
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

            seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(input_embedding_fut)

        transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 25 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)
            
            if generated is not None:
                parent_id = torch.gather(generated, 1, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1,-1,i))
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions
                cache = torch.zeros(input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim, device=input_batch.sem_ids.device)
                cache_mask = torch.cat([torch.ones(input_batch.sem_ids.shape[0], 1, dtype=bool, device=input_batch.seq_mask.device), input_batch.seq_mask], axis=1)
                cache[cache_mask] = self.transformer.cached_enc_output.values()
                lengths = self.transformer.cached_enc_output.offsets().diff().repeat_interleave(k)
                cache = cache.repeat_interleave(k, dim=0)
                self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    @torch.compile
    # TODO: Fix compile for torch padded -> Jagged
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B).sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut[:, 1:].flatten(end_dim=1)
                loss = F.cross_entropy(out, target)
            if not self.training:
                self.transformer.cached_enc_output = None
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None

        return ModelOutput(loss=loss, logits=logits)
