import torch
import torch.nn.functional as F

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
    
    def jagged_forward(self, q: torch.nested.Tensor, k: torch.nested.Tensor, v: torch.nested.Tensor, is_causal: bool) -> torch.nested.Tensor:
        queries = q.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = k.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = v.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        dropout_p = 0. if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=dropout_p, is_causal=is_causal)
        
        context_vec = context_vec.transpose(1, 2).flatten(-2)
        return context_vec

    def forward(self, qkv: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
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

        self.register_buffer("kv_cache", torch.zeros(2, 640, 800, 64, requires_grad=False))
        self.register_buffer("last_cache_pos", torch.zeros(640, 1, requires_grad=False))
        self.cache_empty = True

    def forward(self,
                x: torch.Tensor,
                x_kv: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                jagged: bool = False,
                use_cache: bool = False,
                ) -> torch.Tensor:
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        assert not self.cross_attn or x_kv is not None, "Found null x_kv in cross attn. layer"
        
        if self.cross_attn:
            q, kv = self.q(x), self.kv(x_kv)
            qkv = torch.cat([q, kv], axis=2)
        else:
            qkv = self.qkv(x)
        
        if use_cache and self.cache_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            queries, keys, values = qkv.chunk(3, dim=-1)
            
            self.kv_cache[0][:B, :N, :][padding_mask] = torch.cat(keys.unbind().detach())[:,:]
            self.kv_cache[1][:B, :N, :][padding_mask] = torch.cat(values.unbind().detach())[:,:]
            
            self.last_cache_pos[:B] = padding_mask.sum(axis=-1)
            self.cache_empty = False
        
        elif use_cache and not self.cache_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            queries, keys, values = torch.cat(qkv.unbind()).chunk(3, dim=-1)

            next_token_pos = padding_mask.sum(axis=-1)
            self.kv_cache[0][:B, :N, :][torch.arange(B, device=self.kv_cache.device), self.last_cache_pos.unsqueeze(1)] = keys.detach()[:,:]
            self.kv_cache[1][:B, :N, :][torch.arange(B, device=self.kv_cache.device), self.last_cache_pos.unsqueeze(1)] = values.detach()[:,:]

            keys = torch.nested.nested_tensor(
                [self.kv_cache[0][:B, :N, :][i, :self.last_cache_pos.squeeze()[i]+1] for i in range(B)], 
                layout=torch.jagged,
                device=self.kv_cache.device
            )

            values = torch.nested.nested_tensor(
                [self.kv_cache[1][:B, :N, :][i, :self.last_cache_pos.squeeze()[i]+1] for i in range(B)], 
                layout=torch.jagged,
                device=self.kv_cache.device
            )
        elif jagged:
            queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        if jagged:
            assert attn_mask is None, "Mask not supported by jagged attention"
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=True)
        else:
            context_vec = self.attend(qkv, attn_mask)
    
        context_vec = self.proj(context_vec)
        return context_vec
