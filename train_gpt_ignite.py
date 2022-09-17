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
import logging
from hyperpyyaml import load_hyperpyyaml
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.engine import Events, Engine

from data import TextData
from mingpt import GPT


class BasicLRScheduler:
    '''A simple class for learning rate scheduling, call before iteration.'''
    def __init__(self, base_lr, num_warmup):
        self.num_warmup = num_warmup
        self.base_lr = base_lr
        self.step_count = 0

    def step(self):
        if self.step_count <= self.num_warmup:
            new_value = (self.step_count / self.num_warmup) * self.base_lr
        else:
            offset = self.step_count - self.num_warmup
            new_value = self.base_lr * m.exp(-offset/40000)
        self.step_count += 1
        return new_value


def model_generate(model, sp_model):
    with torch.no_grad():
        model.eval()
        sent = 'Rome was ruled by'
        ids = sp_model.encode(sent)
        inp_sent = torch.tensor(ids, dtype=torch.int64).cuda()
        lst = ids.copy()
        for j in range(30):
            preds = model(inp_sent.view(1, inp_sent.numel()))
            next_word_idx = F.gumbel_softmax(preds[:, -1], hard=True).argmax()
            next_word_idx_ = next_word_idx.cpu().item()
            lst.append(next_word_idx_)
            if '.' in sp_model.id_to_piece(next_word_idx_):
                break
            inp_sent = torch.hstack((inp_sent, next_word_idx,))
        lst = sp_model.decode(lst)
        logger.info(f'Generated sentence: {lst}\n')


class CrossEntLoss(nn.CrossEntropyLoss):
    def __init__(self, weight = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input, target):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        return super().forward(input, target)


class Config:
    def __init__(self, dct):
        self.__dict__.update(dct)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


def main():

    logger.info('Starting script.')
    params = load_hyperpyyaml(open('hparams/gpt.yaml'))
    params = Config(params)
    logger.info(f'Params are:\n{params}')

    params['wd'] = float(params['wd'])
    # wandb.init(project='nnlms', config=params)

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor('wikitext-103/50k_sp.model')
    vocab_size = len(sp_model)

    model = GPT(params)
    print(model)
    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f'Number of parameters: {num_params}')
    bos = sp_model['<s>']
    train_data = TextData('wikitext-103/train.pkl', params['seq_len'], randomize=True)
    valid_data = TextData('wikitext-103/valid.pkl', params['seq_len'])

    logger.info(f'One epoch will take {len(train_data) // params["batch_size"]} iterations')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params['valid_batch_size'],
                                               num_workers=0, shuffle=True)
    logger.info('Finished setting up dataloaders.')
    device = 'cuda'
    model.to(device)
    criterion = CrossEntLoss(reduction='mean')
    optimizer = model.configure_optimizers(params)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = BasicLRScheduler(params['lr'], 3000)
    grad_accum = 8

    def update_step(engine, batch):
        should_step = engine.state.iteration % grad_accum == 0
        model.train()
        x, y = batch[0].cuda(), batch[1].cuda()
        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criterion(pred, y)
        scaler.scale(loss / grad_accum).backward()
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss

    trainer = Engine(update_step)

    evaluator = create_supervised_evaluator(model, {"nll": Loss(criterion)}, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=2000))
    def run_validation(engine):
        evaluator.run(valid_loader)
        valid_nll = evaluator.state.metrics['nll']
        logger.info(f'Iteration {engine.state.iteration} - valid nll {valid_nll:.3f} - ppl {m.exp(valid_nll):.1f}')
        model_generate(model, sp_model)
    
    # @trainer.on(Events.GET_BATCH_COMPLETED(every=100))
    # def print_batch(engine):
    #     x = engine.state.batch[0]
    #     y = engine.state.batch[1]
    #     print(sp_model.decode(x[0].detach().cpu().tolist()))
    #     print(sp_model.decode(y[0].detach().cpu().tolist()))

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Iteration {engine.state.iteration} - nll {engine.state.output:.3f} - lr {lr:.7f}')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
        logger.info(f'Finished epoch, current iteration {engine.state.iteration}')

    @trainer.on(Events.ITERATION_STARTED)
    def update_lr(engine):
        new_lr = lr_scheduler.step()
        optimizer.param_groups[0]['lr'] = new_lr

    trainer.run(train_loader, max_epochs=params['epochs'])


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

