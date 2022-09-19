import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scaling import ScaledLinear, DoubleSwish, BasicNorm
from rotary import RotaryAttention


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.fc1 = ScaledLinear(d_model, d_model*4)
        self.fc2 = ScaledLinear(d_model*4, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dropout, nhead=8, norm_all=True):
        super().__init__()
        self.norm_all = norm_all
        self.attn = RotaryAttention(d_model, 12, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, dropout)
        self.norm = BasicNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(x, attn_mask)
        x = self.norm(x + self.dropout(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size, nhead, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decodelayer = TransformerDecoderLayer(d_model, dropout, nhead=nhead)
        self.layers = [decodelayer]
        for _ in range(num_layers-1):
            self.layers.append(copy.deepcopy(decodelayer))
        self.layers = nn.ModuleList(self.layers)
        self.norm = BasicNorm(d_model)
        self.fc_out = ScaledLinear(d_model, vocab_size)

        for nm, module in self.named_modules():
            if isinstance(module, ScaledLinear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.embedding(x)
        T = x.size(1)
        attn_mask = ~x.new_ones((T, T), dtype=torch.bool).tril_()
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        return self.fc_out(x)
