import argparse
import gin
import torch
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from torch import Tensor
from torch.nested import Tensor as NestedTensor
from torch.autograd import Function


class PaddedToJaggedTensor(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lengths: Tensor, max_len: int):
        mask = torch.arange(max_len, device=x.device).unsqueeze(0).repeat(x.shape[0], 1) < lengths.unsqueeze(1)
        
        ctx.save_for_backward(mask)
        return torch.nested.nested_tensor(
            [i[:j.item()] for i, j in zip(x, lengths)],
            layout=torch.jagged,
            device=x.device,
            requires_grad=x.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_values = grad_output.values()

        grad_x = torch.zeros(*mask.shape, grad_values.shape[-1], dtype=grad_values.dtype, device=grad_values.device)
        grad_x[mask] = grad_values

        return grad_x, None, None


@torch.compiler.disable 
def padded_to_jagged_tensor(x: Tensor, lengths: Tensor, max_len: int) -> NestedTensor:
    """
      Differentiable padded -> Jagged conversion. 
      This will cause a graph break as nested tensor creation is not supported by torch.compile.
    """
    return PaddedToJaggedTensor.apply(x, lengths, max_len)


def jagged_to_flattened_tensor(x: NestedTensor) -> Tensor:
    return x.values()


def reset_kv_cache(fn):
    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out
    
    return inner


def reset_encoder_cache(fn):
    def inner(self, *args, **kwargs):
        self.transformer.cached_enc_output = None
        out = fn(self, *args, **kwargs)
        self.transformer.cached_enc_output = None
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


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)

@torch.no_grad
def compute_debug_metrics(batch: TokenizedSeqBatch) -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    debug_metrics = {
        f"seq_length_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item() 
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    return debug_metrics