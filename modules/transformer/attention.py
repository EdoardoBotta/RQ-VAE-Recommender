import torch
from typing import Optional
from torch import nn


torch.backends.cuda.enable_flash_sdp(True)


class Attend(nn.Module):
    def __init__(self, d_out, num_heads, head_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = dropout

    def forward(self, qkv: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=attn_mask is None)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, cross_attn=False, dropout=0.0, qkv_bias=False) -> None:
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
    
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

        self.attend = Attend(self.d_out, self.num_heads, self.head_dim, self.dropout)

    def forward(self,
                x: torch.Tensor,
                x_kv: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        assert not self.cross_attn or x_kv is not None, "Found null x_kv in cross attn. layer"
        
        if self.cross_attn:
            q, kv = self.q(x), self.kv(x_kv)
            qkv = torch.cat([q, kv], axis=2)
        else:
            qkv = self.qkv(x)

        context_vec = self.attend(qkv, attn_mask)
        context_vec = self.proj(context_vec)

        return context_vec
