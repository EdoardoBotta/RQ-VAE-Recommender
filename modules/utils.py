import torch
from einops import rearrange


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[
        rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices
    ]


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, torch.Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)
