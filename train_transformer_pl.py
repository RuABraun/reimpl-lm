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
from torch.optim.lr_scheduler import _LRScheduler

from torchinfo import summary

from data import TextData
from transformer import Transformer


class LRScheduler(_LRScheduler):
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, num_warmup):
        self.num_warmup = num_warmup
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.num_warmup:
            new_values = [(self._step_count / self.num_warmup) * base_lr for base_lr in self.base_lrs]
            return new_values
        return self.base_lrs


def model_generate(model, sp_model):
    with torch.no_grad():
        model.eval()
        sent = 'Rome was a great city'
        ids = sp_model.encode(sent)
        inp_sent = torch.tensor(ids, dtype=torch.int64).cuda()
        lst = sent.split()
        j = 0
        while j < 30:
            preds = model(inp_sent.view(1, inp_sent.numel()))
            next_word_idx = F.gumbel_softmax(preds[-1], hard=True).argmax()
            next_word_idx_ = next_word_idx.cpu().item()
            lst.append(sp_model.id_to_piece(next_word_idx_))
            if next_word_idx_ == sp_model['.']:
                break
            inp_sent = torch.hstack((inp_sent, next_word_idx,))
            j += 1
        logger.info(f'Generated sentence: {lst}\n')


class TransformerWrapper(pl.LightningModule):
    def __init__(self, model, params, sp_model) -> None:
        super().__init__()
        self.model = model
        self.params = params
        self.sp_model = sp_model
        self.vocab_size = len(sp_model)
        self.hidden = None

    def training_step(self, batch, batch_idx):
        inp_sent, targ_sent = batch
        pred = self.model(inp_sent)
        loss = F.cross_entropy(pred.view(-1, self.vocab_size), targ_sent.view(-1))
        if batch_idx % 100 == 0:
            logger.info(f'Train loss {loss.item():.2f} at step {batch_idx}')
        return loss

    def validation_step(self, batch, batch_idx):
        inp_sent, targ_sent = batch
        pred = self.model(inp_sent)
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
        lr_scheduler = LRScheduler(optimizer, 10000)
        return [optimizer], [lr_scheduler]


def main():

    logger.info('Starting script.')
    params = load_hyperpyyaml(open('transformer.yaml'))
    logger.info(f'Params are:\n{params}')

    params['wd'] = float(params['wd'])
    wandb.init(project='nnlms', config=params)

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor('wikitext-103/50k_sp.model')
    vocab_size = len(sp_model)

    model = Transformer(512, 12, vocab_size)
    summary(model, input_size=(4, 16,), dtypes=[torch.LongTensor], device='cpu')

    train_data = TextData('wikitext-103/train.pkl', params['seq_len'])
    valid_data = TextData('wikitext-103/valid.pkl', params['seq_len'])

    logger.info(f'One epoch will take {len(train_data) // (params["seq_len"]*params["batch_size"])} iterations')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'])
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params['valid_batch_size'],
                                               num_workers=0)
    logger.info('Finished setting up dataloaders.')

    model_wrapper = TransformerWrapper(model, params, sp_model)

    trainer = pl.Trainer(val_check_interval=2000, precision=16, 
        max_steps=100_000, default_root_dir='exp-transformer', accelerator='gpu', accumulate_grad_batches=2)

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
