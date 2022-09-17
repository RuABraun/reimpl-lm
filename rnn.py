import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, use_resblock=False, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_hid = 1024
        if not use_resblock:
            self.linear = nn.Linear(emb_dim, 1024)
        else:
            self.block = ResidualBlock(emb_dim, self.num_hid)
        self.use_resblock = use_resblock
        self.rnn = nn.LSTM(input_size=self.num_hid, hidden_size=self.num_hid, 
            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        N, T, C = x.shape
        x = x.view(N * T, C)
        if not self.use_resblock:
            x = self.linear(x)
        else:
            x = self.block(x)
        out, hidden = self.rnn(x.view(N, T, -1), hidden)
        x = F.relu(self.fc(out.reshape(N * T, -1)))
        x = self.classifier(x)
        return x, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.num_hid),
                    weight.new_zeros(1, bsz, self.num_hid))


class ResidualBlock(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        z = F.layer_norm(self.linear2(F.relu(self.linear1(x))), (self.out_dim,))
        if z.size(-1) > x.size(-1):
            x = F.pad(x, (0, z.size(-1) - x.size(-1)))
            return z + x
        else:
            return z + x