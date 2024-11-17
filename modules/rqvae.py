import torch

from torch import nn
from typing import List
from typing import NamedTuple

from data.schemas import SeqBatch
from .encoder import MLP
from .loss import ReconstructionLoss
from .loss import RqVaeLoss
from .quantize import Quantize
from init.kmeans import kmeans_init_


class RqVaeOutput(NamedTuple):
    embeddings: torch.Tensor
    residuals: torch.Tensor
    sem_ids: torch.Tensor


class RqVae(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        commitment_weight: float = 0.25
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.layers = nn.ModuleList(modules=[
            Quantize(embed_dim=embed_dim, n_embed=codebook_size)
            for _ in range(n_layers)
        ])

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims,
            out_dim=input_dim
        )

        self.reconstruction_loss = ReconstructionLoss()
        self.rqvae_loss = RqVaeLoss(self.commitment_weight)

    def kmeans_init(self, batch: SeqBatch) -> None:
        x = batch.x
        with torch.no_grad():
            x = self.encoder(x)
            for layer in self.layers:
                kmeans_init_(layer.weight, x=x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def get_semantic_ids(self,
                         x: torch.Tensor,
                         gumbel_t: float = 0.001) -> RqVaeOutput:
        res = self.encode(x)
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=torch.stack(embs, dim=-1),
            residuals=torch.stack(residuals, dim=-1),
            sem_ids=torch.stack(sem_ids, dim=-1)
        )

    def forward(self, batch: SeqBatch, gumbel_t: float) -> torch.Tensor:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=-1))

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = self.rqvae_loss(residuals, embs)
        loss = (reconstuction_loss + rqvae_loss).mean()

        return loss
