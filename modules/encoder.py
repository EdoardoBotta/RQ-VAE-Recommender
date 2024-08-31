import torch
from torch import nn
from torch.nn import functional as F


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        normalize: bool = False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            L2NormalizationLayer() if normalize else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)
