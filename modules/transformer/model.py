from modules.encoder import MLP
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.attention import MultiHeadAttention
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor


class KVCacheOpsMixin:
    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()
    
    def apply_to_kv_cache(self, fn) -> None:
        for layer in self.layers:
            layer.apply_to_kv_cache(fn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden_dims: List[int] = [1024],
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn
        self.enable_kv_cache = enable_kv_cache

        self.attention = MultiHeadAttention(
            d_in=d_in, d_out=d_out, num_heads=num_heads, cross_attn=False, dropout=dropout, qkv_bias=qkv_bias, enable_kv_cache=enable_kv_cache
        )

        self.ff = nn.Sequential(
            RMSNorm(d_out),
            MLP(
                input_dim=d_out,
                hidden_dims=mlp_hidden_dims,
                out_dim=d_out,
                dropout=dropout,
                normalize=False
            ),
            nn.Dropout(dropout)
        )

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        self.do = nn.Dropout(dropout)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadAttention(
                d_in=d_out, d_out=d_out, num_heads=num_heads, cross_attn=True, dropout=dropout, qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out)
    
    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: Optional[bool] = False
    ) -> AttentionInput:
        attn_out = x + self.attention(self.do(self.attn_norm(x)), padding_mask=padding_mask, is_causal=is_causal, jagged=jagged, use_cache=not self.training and self.enable_kv_cache)
        if self.do_cross_attn:
            attn_out = attn_out + self.cross_attention(
                x=self.do(self.cross_attn_norm(x)), x_kv=x_kv, padding_mask=padding_mask, is_causal=False, jagged=jagged, use_cache=not self.training and self.enable_kv_cache
            )
        proj_out = attn_out + self.ff(attn_out)
        return proj_out
    
    def reset_kv_cache(self):
        self.attention.kv_cache.reset()
        if self.do_cross_attn:
            self.cross_attention.kv_cache.reset()

    def apply_to_kv_cache(self, fn):
        self.attention.kv_cache.apply(fn)
        if self.do_cross_attn:
            self.cross_attention.kv_cache.apply(fn)


class TransformerDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()

        self.do_cross_attn = do_cross_attn

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                num_heads=num_heads,
                qkv_bias=False,
                do_cross_attn=self.do_cross_attn,
                enable_kv_cache=enable_kv_cache
            ) for _ in range(n_layers)
        ])

    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        for layer in self.layers:
            x = layer(x=x, x_kv=context, padding_mask=padding_mask, is_causal=is_causal, jagged=jagged)
        return x
    
    @property
    def seq_lengths(self) -> Tensor:
        return self.layers[0].attention.kv_cache.seq_lengths


class TransformerEncoderDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()

        self.encoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=encoder_layers,
            do_cross_attn=False,
            enable_kv_cache=False
        )

        self.decoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=decoder_layers,
            do_cross_attn=True,
            enable_kv_cache=False
        )

        self.layers = [self.encoder, self.decoder]
        self.cached_enc_output = None
    
    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        if self.cached_enc_output is None:
            context = self.encoder(context, padding_mask=padding_mask, is_causal=False, context=None, jagged=jagged)
            if not self.training:
                self.cached_enc_output = context
        else:
            context = self.cached_enc_output
        out = self.decoder(x, padding_mask=None, is_causal=True, context=context, jagged=jagged)
        return out
        
