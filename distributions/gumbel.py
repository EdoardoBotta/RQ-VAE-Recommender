import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple


def sample_gumbel(shape: Tuple, device: torch.device, eps=1e-20) -> Tensor:
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: Tensor, temperature: float, device: torch.device) -> Tensor:
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample


class TemperatureScheduler:
    def __init__(
        self,
        t0: float,
        min_t: float,
        anneal_rate: float,
        step_size: int,
    ) -> None:
        self.t0 = t0
        self.min_t = min_t
        self.anneal_rate = anneal_rate
        self.step_size = step_size
        self.t = t0

    def update_t(self, iter):
        if iter % self.step_size == self.step_size-1:
            self.t = np.maximum(self.t*np.exp(-self.anneal_rate*iter), self.min_t)

    def get_t(self, iter):
        self.update_t(iter)
        return self.t
