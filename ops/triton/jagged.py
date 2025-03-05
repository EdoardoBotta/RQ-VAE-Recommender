import torch
import triton
import triton.language as tl

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


def padded_to_jagged(x, lengths, target):
    assert x.dim() == 3
    assert lengths.shape[0] == x.shape[0]
    assert x.is_contiguous()

    B, N, D = x.shape

    offsets = torch.cat([
        torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
        lengths.cumsum(dim=0)
    ])
    grid = lambda meta: (B*triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(D, meta['BLOCK_SIZE_D']),)

    _padded_to_jagged_kernel[grid](
        x, lengths, offsets,
        target._values, target._offsets,
        x.stride(0), x.stride(1), x.stride(2), target._values.stride(0),
        B, N, D, BLOCK_SIZE_N=4, BLOCK_SIZE_D=D
    )

    return target