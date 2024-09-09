import torch
from typing import List
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
        hidden_dims: List[int],
        out_dim: int,
        normalize: bool = False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]
        
        modules = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            modules.append(nn.Linear(in_d, out_d))
            modules.append(nn.ReLU())
        modules.append(L2NormalizationLayer() if normalize else nn.Identity())

        self.mlp = nn.Sequential(nn.ModuleList(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)
