import torch

from ..rqvae import RqVae
from ..rqvae import RqVaeOutput
from ...data.schemas import SeqBatch
from typing import NamedTuple
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TokenizedSeqBatch(NamedTuple):
    user_ids: torch.Tensor
    sem_ids: torch.Tensor
    seq_mask: torch.Tensor
    

class SemanticIdTokenizer(nn.Module):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 codebook_size: int,
                 n_layers: int = 3,
                 commitment_weight: float = 0.25) -> None:
        super().__init__()

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            n_layers=n_layers,
            commitment_weight=commitment_weight
        )

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.cached_ids = None
    
    def _get_hits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return (key.unsqueeze(0) == query.unsqueeze(1)).all(axis=-1)
    
    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self.rq_vae(x, t)
    
    def precompute_corpus_ids(self, movie_dataset: Dataset) -> torch.Tensor:
        cached_ids = None
        dedup_dim = []
        for batch in DataLoader(movie_dataset, batch_size=128, shuffle=False):
            batch_ids = self.tokenize(batch)
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids).all(axis=-1)
            hits = torch.triu(is_hit, diagonal=1).sum(axis=-1) - 1
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids).all(axis=-1)
                hits += torch.triu(is_hit, diagonal=1).sum(axis=-1) - 1
                cached_ids = torch.cat([cached_ids, batch_ids], axis=0)
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids
        dedup_dim_tensor = torch.cat(dedup_dim).unsqueeze(-1)
        self.cached_ids = torch.cat([
            cached_ids,
            dedup_dim_tensor], axis=-1)
        return self.cached_ids
    
    def tokenize(self, batch: SeqBatch) -> RqVaeOutput:
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            self.eval()
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).reshape(B, N*self.n_layers)
        else:
            B, N = batch.ids.shape
            sem_ids = self.cached_ids[batch.ids.flatten(), :].reshape(B, N*self.n_layers)
        assert sem_ids.shape[0] == B and sem_ids.shape[1] == N*self.n_layers, f"Invalid sem ids shape, expected {(B,N*self.n_layers)}, found {sem_ids.shape}"
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            seq_mask=None  # TODO: Fix this by repeating seq_mask from batch
        )