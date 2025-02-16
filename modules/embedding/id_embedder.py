import torch

from data.schemas import TokenizedSeqBatch
from torch import nn
from torch import Tensor
from typing import NamedTuple


class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim*num_embeddings
        
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings*self.sem_ids_dim+1,
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(self, batch: TokenizedSeqBatch) -> Tensor:
        sem_ids = batch.token_type_ids*self.num_embeddings + batch.sem_ids
        sem_ids[~batch.seq_mask] = self.padding_idx

        if batch.sem_ids_fut is not None:
            sem_ids_fut = batch.token_type_ids_fut*self.num_embeddings + batch.sem_ids_fut
            sem_ids_fut = self.emb(sem_ids_fut)
        else:
            sem_ids_fut = None
        return SemIdEmbeddingBatch(
            seq=self.emb(sem_ids),
            fut=sem_ids_fut
        ) 
    

class UserIdEmbedder(nn.Module):
    # TODO: Implement hashing trick embedding for user id
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        hashed_indices = x % self.num_buckets
        # hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)
