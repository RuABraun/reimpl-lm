from typing import Tuple
import math

import torch
import torch.nn as nn

from einops import rearrange, repeat
from scaling import ScaledLinear
from flash_attn.flash_attention import FlashAttention


def rotate_half(x):
    # rearrange doesn't work with torch.jit
    # x = rearrange(x, '... (d r) -> ... d r', r=2)
    x = x.unflatten(dim=-1, sizes=(-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack((-x2, x1), dim=-1)
    # return rearrange(rotated_x, '... d r -> ... (d r)')
    return rotated_x.flatten(start_dim=-2)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int = -2):
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:x.shape[seq_dimension], :]
    sin = sin[:x.shape[seq_dimension], :]
    if seq_dimension == -3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.Wqkv = ScaledLinear(d_model, d_model*3)
        self.out_proj = ScaledLinear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attention_dropout = dropout
        self.inner_attn = FlashAttention()

    def forward(self, x, attn_mask):
        qkv = self.Wqkv(x)
        query, key, value = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3,
                                          h=self.num_heads).unbind(dim=2)
        query, key = self.rotary_emb(query, key, seq_dimension=-3)
        qkv = torch.stack([query, key, value], dim=2)
        if qkv.dtype == torch.float16:
            context, attn_weights = self.inner_attn(qkv, causal=True)
        else:
            bs = query.size(0)
            seqlen = query.size(1)
            query = query.transpose(1, 2).reshape(bs*self.num_heads, seqlen, self.head_dim)
            key = key.transpose(1, 2).reshape(bs*self.num_heads, seqlen, self.head_dim)
            attn_weights = torch.bmm(query, key.transpose(1, 2)) / self.head_dim ** 0.5 # -> b*h s s
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
            attn_weights = attn_weights.view(bs, self.num_heads, seqlen, seqlen)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            # b h s s
            attn_weights = attn_weights.view(bs*self.num_heads, seqlen, seqlen)
            # b*h s s
            attn_weights = nn.functional.dropout(attn_weights, self.attention_dropout, training=self.training)

            value = value.transpose(1, 2).reshape(bs * self.num_heads, seqlen, self.head_dim)
            context = torch.bmm(attn_weights, value)  # -> b*h s d
            context = context.view(bs, self.num_heads, seqlen, self.head_dim).transpose(1, 2)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seq_len != self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = repeat(torch.cos(freqs).to(x.dtype), '... d -> ... (d 2)')
            self._sin_cached = repeat(torch.sin(freqs).to(x.dtype), '... d -> ... (d 2)')

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                seq_dimension=-2) -> Tuple[torch.Tensor, torch.Tensor]:
        assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=seq_dimension
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension),
        )