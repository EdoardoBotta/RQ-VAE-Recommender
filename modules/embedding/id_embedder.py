import torch
from torch import nn


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embeddings_dim, padding_idx=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)
    

class UserIdEmbedder(nn.Module):
    # TODO: Implement hashing trick embedding for user id
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)
