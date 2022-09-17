import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rotary import RotaryEmbedding


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2048)
        self.fc2 = nn.Linear(2048, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class RotaryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(d_model, d_model*3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attention_dropout = dropout

    def forward(self, x, attn_mask):
        qkv = self.Wqkv(x)
        query, key, value = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3,
                                          h=self.num_heads).unbind(dim=2)
        query, key = self.rotary_emb(query, key, seq_dimension=-3)
        qkv = torch.stack([query, key, value], dim=2)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = RotaryAttention(d_model, num_heads=8, dropout=0.)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model)
        self.dropout = nn.Dropout(p=0.)
        
    def forward(self, x, attn_mask):
        y = self.norm1(x)
        x = x + self.attn(y, attn_mask=attn_mask)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decodelayer = TransformerDecoderLayer(d_model)
        self.layers = [decodelayer]
        for _ in range(num_layers-1):
            self.layers.append(copy.deepcopy(decodelayer))
        self.layers = nn.ModuleList(self.layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        for nm, module in self.named_modules():
            if isinstance(module, nn.Linear):
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

    def configure_optimizers(self, params):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": params['wd']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=params['lr'], betas=(0.9, 0.95))
        return optimizer
        