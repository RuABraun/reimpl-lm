import torch
from tqdm import tqdm
import pickle
import torch.nn as nn
from loguru import logger
from torch.utils.data import IterableDataset, Dataset
import random
import fire

SB = '<sb>'
UNK = '<unk>'


class TextDataIterable(IterableDataset):
    def __init__(self, pickle_f, seq_len, batch_size=None):
        super().__init__()

        with open(pickle_f, 'rb') as fh:
            text = pickle.load(fh)
        self.text = torch.tensor(text, dtype=torch.int64)
        self.seq_len = seq_len
        self.start_index = 0
        self.end_index = self.text.size(0)
        self.batch_size = batch_size
        logger.info(f'Loaded text of size {self.end_index}')

    def __len__(self):
        return self.end_index - self.start_index

    def __iter__(self):
        size = self.text.size(0)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            size_per_worker = size // num_workers
            self.start_index = worker_id * size_per_worker
            self.end_index = (worker_id + 1) * size_per_worker
        if self.batch_size is None:
            self.iter = iter(range(self.start_index, self.end_index - self.seq_len, self.seq_len))
        else:
            l = int(len(self) / self.seq_len / self.batch_size) * self.seq_len * self.batch_size
            indices = torch.arange(self.start_index, self.start_index + l, self.seq_len).reshape(self.batch_size, l // self.batch_size // self.seq_len).transpose(1, 0).ravel()
            self.iter = iter(indices)

        return self

    def __next__(self):
        index = next(self.iter)
        sentence = self.text[index: index + self.seq_len]
        return sentence[:-1], sentence[1:]


class TextData(Dataset):
    def __init__(self, pickle_f, seq_len, randomize=False):
        super().__init__()

        with open(pickle_f, 'rb') as fh:
            text = pickle.load(fh)
        self.randomize = randomize
        self.text = torch.tensor(text, dtype=torch.int64)
        self.seq_len = seq_len
        self.start_index = 0
        self.end_index = self.text.size(0)
        logger.info(f'Loaded text of size {self.end_index}')

    def __len__(self):
        return (self.end_index - self.start_index - self.seq_len) // self.seq_len - 1

    def __getitem__(self, index):
        #offset = random.randint(0, self.seq_len - 1) if self.randomize else 0
        index = index * self.seq_len
        x = self.text[index: index + self.seq_len + 1]
        return x[:-1], x[1:]

class SymTab:
    def __init__(self, syms):
        self.syms = sorted(set(syms))
        mapping = {}
        for i, word in enumerate(self.syms):
            mapping[i] = word
            mapping[word] = i
        if UNK not in mapping:
            mapping[i+1] = UNK
            mapping[UNK] = i + 1
            self.syms.append(UNK)
        self.mapping = mapping

    def __len__(self):
        return len(self.syms)

    def __getitem__(self, key):
        return self.mapping[key]

    def get(self, key, default):
        return self.mapping.get(key, default)

    def stringify(self, tens):
        if tens.ndim == 1:
            tens = tens.view(1, -1)
        lines = []
        for i in range(tens.size(0)):
            vals = tens[i].tolist()
            s = []
            for num in vals:
                s.append(self[num])
            lines.append(' '.join(s))
        return '\n'.join(lines)


def load_vocab(vocab_f):
    vocab = []
    for line in open(vocab_f):
        vocab.append(line.split()[0])
    symtab = SymTab(vocab)
    return symtab


def create_binarized_text(vocab_f, text_f, outf):
    mapping = load_vocab(vocab_f)
    sb_index = mapping[SB]
    unk_index = mapping[UNK]
    text = [sb_index]
    oov_count = 0
    for line in tqdm(open(text_f)):
        if not line.strip() or not line.endswith(f'{SB}\n'):
            continue
        for w in line.split():
            idx = mapping.get(w, unk_index)
            if idx == unk_index:
                oov_count += 1
            text.append(idx)
    logger.info(f'Number of OOVs is {oov_count} - percent is {100.*oov_count/len(text)}%')

    with open(outf, 'wb') as fh:
        pickle.dump(text, fh)


def create_binarized_text_bpe(bpe_f, text_f, outf):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(bpe_f)
    sb_index = sp['.']
    unk_index = sp[UNK]
    text = [sb_index]
    oov_count = 0
    for line in tqdm(open(text_f)):
        if not line.strip():
            continue
        ids = sp.encode(line)
        for id in ids:
            if id == unk_index:
                oov_count += 1
        text.extend(ids)
    logger.info(f'Number of OOVs is {oov_count} - percent is {100.*oov_count/len(text)}%')

    with open(outf, 'wb') as fh:
        pickle.dump(text, fh)


if __name__ == '__main__':
    import plac;
    plac.call(create_binarized_text_bpe)
