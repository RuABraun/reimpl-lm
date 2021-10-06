import torch
import pickle
import torch.nn as nn
from torch.utils.data import IterableDataset
import fire

SB = '<s>'
UNK = '<unk>'


class TextData(IterableDataset):
    def __init__(self, pickle_f, seq_len):
        super(IterableDataset).__init__()

        with open(pickle_f, 'rb') as fh:
            text = pickle.load(fh)
        self.text = torch.tensor(text, dtype=torch.int32)
        self.seq_len = seq_len
        self.start_index = 0
        self.end_index = self.text.size()

    def __len__(self):
        return len(self.end_index - self.start_index)

    def __iter__(self):
        size = self.text.size()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            size_per_worker = size // num_workers
            self.start_index = worker_id * size_per_worker
            self.end_index = (worker_id + 1) * size_per_worker
        return self

    def __next__(self):
        for index in range(self.start_index, self.end_index, self.seq_len):
            yield self.text[index: index + self.seq_len]


def create_binarized_text(vocab, text_f, outf):
    vocab = sorted(vocab)
    mapping = {}
    for i, word in enumerate(vocab):
        mapping[i] = word
        mapping[word] = i
    mapping[i+1] = UNK
    mapping[UNK] = i + 1

    sb_index = mapping[SB]
    unk_index = mapping[UNK]
    text = []
    for line in open(text_f):
        text.append(sb_index)
        for w in line.split():
            idx = mapping.get(w, unk_index)
            text.append(idx)
    text.append(sb_index)

    with open(outf, 'wb') as fh:
        pickle.dump(text, fh)


if __name__ == '__main__':
    fire.Fire()
