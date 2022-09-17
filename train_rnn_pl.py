#!/usr/bin/env python
import torch
import wandb
import re
import sys
import os
print(os.getcwd())

import math as m
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import pytorch_lightning as pl

from torchinfo import summary

from data import TextDataIterable, load_vocab, SB
from rnn import RnnLM


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def model_generate(model, sp_model):
    with torch.no_grad():
        model.eval()
        inp_sent = torch.tensor(sp_model['.'], dtype=torch.int64).cuda()
        lst = ['.']
        j = 0
        while j < 30:
            preds, _ = model(inp_sent.view(1, j+1))
            next_word_idx = F.gumbel_softmax(preds[-1], hard=True).argmax()
            next_word_idx_ = next_word_idx.cpu().item()
            lst.append(sp_model.id_to_piece(next_word_idx_))
            if next_word_idx_ == sp_model['.']:
                break
            inp_sent = torch.hstack((inp_sent, next_word_idx,))
            j += 1
        logger.info(f'Generated sentence: {lst[1:]}\n')


class RnnWrapper(pl.LightningModule):
    def __init__(self, model, params, sp_model) -> None:
        super().__init__()
        self.model = model
        self.params = params
        self.sp_model = sp_model
        self.vocab_size = len(sp_model)
        self.hidden = None

    def training_step(self, batch, batch_idx):
        if self.hidden is None:
            self.hidden = self.model.init_hidden(self.params["batch_size"])
        self.hidden = repackage_hidden(self.hidden)
        inp_sent, targ_sent = batch
        pred, self.hidden = self.model(inp_sent, self.hidden)

        loss = F.cross_entropy(pred.view(-1, self.vocab_size), targ_sent.view(-1))
        if batch_idx % 100 == 0:
            logger.info(f'Train loss {loss.item():.2f} at step {batch_idx}')
        return loss

    def training_epoch_end(self, outputs):
        self.hidden = self.model.init_hidden(self.params["batch_size"])

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            logger.info('Inside!')
            self.hidden_valid = self.model.init_hidden(self.params["valid_batch_size"])
        else:
            self.hidden_valid = repackage_hidden(self.hidden_valid)
        inp_sent, targ_sent = batch
        pred, self.hidden_valid = self.model(inp_sent, self.hidden_valid)
        loss = F.cross_entropy(pred.view(-1, self.vocab_size), targ_sent.view(-1))
        return {"valid_loss": loss}
        
    def validation_epoch_end(self, losses):
        if not losses:
            return
        valid_loss = sum(l["valid_loss"].item() for l in losses) / len(losses)
        logger.info(f'Valid loss {valid_loss:.2f} ppl {m.exp(valid_loss)}')
        model_generate(self.model, self.sp_model)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.params['lr'], 
            params=self.parameters(), weight_decay=self.params['wd'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            factor=0.1, patience=1, min_lr=self.params['lr'] * 0.1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, 
            "monitor": "valid_loss"}


def main():

    logger.info('Starting script.')
    params = load_hyperpyyaml(open('rnn.yaml'))
    logger.info(f'Params are:\n{params}')

    params['wd'] = float(params['wd'])
    wandb.init(project='rnnlm', config=params)

    cross_batch_context = params['cross_batch_context']

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor('wikitext-103/spm_50k_.model')
    vocab_size = len(sp_model)

    model = RnnLM(vocab_size, 256, use_resblock=params['use_resblock'], dropout=params['dropout'])
    summary(model, input_size=(4, 16,), dtypes=[torch.LongTensor], device='cpu')

    train_data = TextDataIterable('wikitext-103/train.pkl', params['seq_len'], params['batch_size'] if cross_batch_context else None)
    valid_data = TextDataIterable('wikitext-103/valid.pkl', 256, params['valid_batch_size'])

    logger.info(f'One epoch will take {len(train_data) // (params["seq_len"]*params["batch_size"])} iterations')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'])
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params['valid_batch_size'],
                                               num_workers=0)
    logger.info('Finished setting up dataloaders.')

    model_wrapper = RnnWrapper(model, params, sp_model)

    trainer = pl.Trainer(val_check_interval=2000, precision=16, 
        max_steps=100_000, default_root_dir='exp-rnn', accelerator='gpu')

    trainer.fit(model=model_wrapper, train_dataloaders=train_loader, 
        val_dataloaders=valid_loader)
        

def rms(x):
    return torch.mean(x**2)**0.5


def report_gradients(step, module):
    found_grad = False
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is not None and param.ndim >= 2:
            found_grad = True
            # std = torch.std(param)
            norm = rms(param)
            # grad_std = torch.std(grad)
            grad_norm = rms(grad)
            data = {f"grad/{name}-rms": grad_norm.item(),
                f"param/{name}-rms": norm.item(),
                }
            wandb.log(data=data, step=step)
    if not found_grad:
        logger.info('Not found gradient.')


if __name__ == '__main__':
    main()
