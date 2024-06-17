import torch

from distributions.gumbel import gumbel_softmax_sample
from torch import nn
from typing import Tuple

class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self._init_weights()

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.2)
    
    def get_item_embeddings(self, item_ids) -> torch.Tensor:
        return self.embedding(item_ids)
    
    def forward(self, x, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.embed_dim, f"Invalid input dim: Expected {self.embed_dim}, found {x.shape[-1]}"

        codebook = self.embedding.weight
        dist = (
            (x**2).sum(axis=1, keepdim=True) +
            (codebook.T**2).sum(axis=0, keepdim=True) -
            2 * x @ codebook.T
        ) 

        _, ids = (-dist).max(axis=1)

        if self.train:
            weights = gumbel_softmax_sample(-dist, temperature=temperature, device=self.device)
            emb = weights @ codebook
        else:
            emb = self.get_item_embeddings(ids)
        
        return {
            "embeddings": emb,
            "ids": ids
        }