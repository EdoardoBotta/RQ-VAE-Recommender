import torch
import triton
import triton.language as tl

from torch import Tensor
from torch.autograd import Function
from torch.nested import Tensor as NestedTensor


class PaddedToJaggedTensor(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lengths: Tensor, max_len: int) -> NestedTensor:
        assert x.dim() == 3
        assert lengths.shape[0] == x.shape[0]
        assert x.is_contiguous()

        B, N, D = x.shape
        mask = torch.arange(max_len, device=x.device).unsqueeze(0).repeat(x.shape[0], 1) < lengths.unsqueeze(1)
        ctx.save_for_backward(mask)
        lengths = lengths.to(torch.int32)

        # Previous version (breaks compile graph): 
        # return torch.nested.nested_tensor(
        #    [i[:j.item()] for i, j in zip(x, lengths)],
        #    layout=torch.jagged,
        #    device=x.device,
        #    requires_grad=x.requires_grad
        #)

        offsets = torch.cat([
            torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
            lengths.cumsum(dim=0)
        ])

        jagged_batch_size = lengths.sum().to(torch.int32)

        # Initialize empty tensor with right shapes
        target = torch.nested.nested_tensor(
            [[]],
            layout=torch.jagged,
            device=x.device,
            requires_grad=x.requires_grad
        )
        target._size = (B, target.shape[1], D)
        target._strides = (D*target._strides[0], D, 1)
        target._values = torch.empty(jagged_batch_size, D, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        target._offsets = torch.empty(len(lengths)+1, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        target._metadata_cache = {}

        grid = lambda meta: (B*triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(D, meta['BLOCK_SIZE_D']),)

        _padded_to_jagged_kernel[grid](
            x, lengths, offsets,
            target._values, target._offsets,
            x.stride(0), x.stride(1), x.stride(2), target._values.stride(0),
            B, N, D, BLOCK_SIZE_N=32, BLOCK_SIZE_D=D
        )

        # Hack: Fixes autograd failure:
        target._get_max_seqlen()
        target._get_min_seqlen()

        # Hack: Fixes strides after _size change.
        # Doing it here with noop to avoid shape mismatch later
        target = target + 1 - 1
        return target
    

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_values = grad_output.values()

        grad_x = torch.zeros(*mask.shape, grad_values.shape[-1], dtype=grad_values.dtype, device=grad_values.device)
        grad_x[mask] = grad_values

        return grad_x, None, None


def padded_to_jagged_tensor(x: Tensor, lengths: Tensor, max_len: int) -> NestedTensor:
    """
      Differentiable padded -> Jagged conversion. 
      This will cause a graph break as nested tensor creation is not supported by torch.compile.
    """
    return PaddedToJaggedTensor.apply(x, lengths, max_len)


def jagged_to_flattened_tensor(x: NestedTensor) -> Tensor:
    return x.values()


@triton.jit
def _padded_to_jagged_kernel(
    x_ptr,
    lengths_ptr,
    offsets_ptr,
    out_values_ptr,
    out_offsets_ptr,
    x_stride_B, x_stride_N, x_stride_D,
    out_values_stride_B,
    B, N, D,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    assert BLOCK_SIZE_D == D
    pid_n = tl.program_id(0)
    num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)

    row_group = pid_n // num_pids_n
    col_group = pid_n % num_pids_n

    jagged_row_start_offset, jagged_row_end_offset = tl.load(offsets_ptr + row_group), tl.load(offsets_ptr + row_group + 1)
    padded_tile_start_ptr = x_ptr + row_group*x_stride_B + col_group*BLOCK_SIZE_N*x_stride_N
    padded_tile_offsets = tl.arange(0, end=BLOCK_SIZE_N*BLOCK_SIZE_D)

    max_offset_B = (jagged_row_end_offset - jagged_row_start_offset - col_group*BLOCK_SIZE_N)*x_stride_N
    mask = padded_tile_offsets < max_offset_B
    padded_in_ptr = padded_tile_start_ptr + padded_tile_offsets
    in_values = tl.load(padded_in_ptr, mask=mask)

    out_values_tile_start = out_values_ptr + (jagged_row_start_offset + col_group*BLOCK_SIZE_N)*out_values_stride_B
    out_values_ptr = out_values_tile_start + tl.arange(0, BLOCK_SIZE_N*BLOCK_SIZE_D)
    
    tl.store(out_values_ptr, in_values, mask=mask)
    tl.store(out_offsets_ptr + row_group + tl.arange(0, 2), tl.join(jagged_row_start_offset, jagged_row_end_offset))
