import torch
import torch.nn as nn


class RnnLM(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, 1024)
        self.rnn = nn.LSTM(input_size=1024, hidden_size=1024, proj_size=512, batch_first=True)
        self.classifier = nn.Linear(512, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        N, T, C = x.shape
        x = x.view(N * T, C)
        x = self.linear(x)
        out, _ = self.rnn(x.view(N, T, self.linear.weight.size(1)))
        x = self.classifier(x.view(N, T, -1))
        return x