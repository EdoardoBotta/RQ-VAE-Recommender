import torch
from einops import rearrange
from torch import Tensor
from torch.nested import Tensor as NestedTensor


def reset_kv_cache(fn):
    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out
    
    return inner


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: Tensor, indices: Tensor) -> torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[
        rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices
    ]


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


@torch.compiler.disable
def padded_to_jagged_tensor(x: Tensor, lengths: Tensor) -> NestedTensor:
    return torch.nested.nested_tensor(
        [i[:j.item()] for i, j in zip(x, lengths)],
        layout=torch.jagged,
        device=x.device
    )


def jagged_to_flattened_tensor(x: NestedTensor) -> Tensor:
    return x.values()
