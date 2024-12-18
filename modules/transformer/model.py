import torch
from .attention import MultiHeadSelfAttention
from .attention import MultiHeadCrossAttention
from ..normalize import RMSNorm
from typing import Optional
from torch import nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 dropout: float,
                 num_heads: int,
                 qkv_bias: bool,
                 do_cross_attn: bool = False) -> None:
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

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadCrossAttention(
                d_in=d_in, d_out=d_out, num_heads=num_heads, dropout=dropout, qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out)
    
    def forward(self, x: torch.Tensor, x_kv: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn_norm(x + self.attention(x))
        if self.do_cross_attn:
            attn_out = self.cross_attn_norm(attn_out + self.cross_attention(x_q=x, x_kv=x_kv, attn_mask=attn_mask))
        proj_out = self.ffn_norm(attn_out + self.ff(attn_out))
        return proj_out


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 dropout: float,
                 num_heads: int,
                 n_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                num_heads=num_heads,
                qkv_bias=False,
                do_cross_attn=False
            ) for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, attn_mask=attn_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 dropout: float,
                 num_heads: int,
                 n_layers: int,
                 do_cross_attn: bool = False) -> None:
        super().__init__()

        self.do_cross_attn = do_cross_attn

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                num_heads=num_heads,
                qkv_bias=False,
                do_cross_attn=self.do_cross_attn
            ) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, context)
        return x
