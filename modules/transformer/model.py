import torch
from .attention import MultiHeadSelfAttention
from .attention import MultiHeadCrossAttention
from typing import Optional
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 dropout,
                 num_heads,
                 qkv_bias,
                 norm_eps,
                 do_cross_attn=False) -> None:
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn

        self.attention = MultiHeadSelfAttention(
            d_in=d_in, d_out=d_out, num_heads=num_heads, dropout=dropout, qkv_bias=qkv_bias
        )

        self.ff = nn.Linear(d_out, d_out, bias=False)

        self.attn_norm = RMSNorm(d_out, norm_eps=norm_eps)
        self.ffn_norm = RMSNorm(d_out, norm_eps=norm_eps)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadCrossAttention(
                d_in=d_in, d_out=d_out, num_heads=num_heads, dropout=dropout, qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out, norm_eps=norm_eps)
    
    def forward(self, x: torch.Tensor, x_kv: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn_norm(x + self.attention(x))
        if self.do_cross_attn:
            attn_out = self.cross_attn_norm(attn_out + self.cross_attention(x_q=x, x_kv=x_kv, attn_mask=attn_mask))
        proj_out = self.ffn_norm(attn_out + self.ff(attn_out))
        return proj_out

