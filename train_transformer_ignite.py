#!/usr/bin/env python
import torch
import wandb
import re
import sys
import os

import math as m
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint, global_step_from_engine

from data import TextData
from transformer import Transformer


class BasicLRScheduler:
    '''A simple class for learning rate scheduling, call before iteration.'''
    def __init__(self, base_lr, num_warmup):
        self.num_warmup = int(num_warmup)
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

    def state_dict(self):
        return {'num_warmup': self.num_warmup, 'base_lr': self.base_lr, 'step_count': self.step_count}

    def load_state_dict(self, dct):
        self.num_warmup = dct['num_warmup']
        self.base_lr = dct['base_lr']
        self.step_count = dct['step_count']

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


def parse_args():
    args = sys.argv[1:]
    args_dct = {arg.split('=')[0]: arg.split('=')[1] for arg in args}
    for key, value in args_dct.items():
        if re.match('^[0-9]+$', value):
            args_dct[key] = int(value)
        elif re.match('^[0-9.]+$', value):
            args_dct[key] = float(value)
    return args_dct

def main():

    logger.info('Starting script.')
    args_dct = parse_args()
    params = load_hyperpyyaml(open('hparams/transformer.yaml'), overrides=args_dct)
    params['wd'] = float(params['wd'])
    params['lr'] = float(params['lr'])
    logger.info(f'Params are:\n{params}')

    wandb.init(project='nnlms', config=params, group='transformer')

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor('wikitext-103/50k_sp.model')
    vocab_size = len(sp_model)

    model = Transformer(512, 12, vocab_size)
    print(model)
    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f'Number of parameters: {num_params}')

    train_data = TextData('wikitext-103/train.pkl', params['seq_len'], randomize=True)
    valid_data = TextData('wikitext-103/valid.pkl', params['seq_len'])

    logger.info(f'One epoch will take {len(train_data) // params["batch_size"]} iterations')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'], drop_last=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params['valid_batch_size'],
                                               num_workers=0, drop_last=True, shuffle=True)
    logger.info('Finished setting up dataloaders.')
    device = 'cuda'
    model.to(device)
    criterion = CrossEntLoss(reduction='mean')
    optimizer = model.configure_optimizers(params)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = BasicLRScheduler(params['lr'], params['warmup_updates'])
    grad_accum = params['grad_accum']

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
            if engine.state.iteration % 20 == 0 and engine.state.iteration > 1:
                report_gradients(engine.state.iteration, model, optimizer)
            if params['clip_norm'] != 0.:
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
        logger.info(f'Iteration {engine.state.iteration/grad_accum} - valid nll {valid_nll:.3f} - ppl {m.exp(valid_nll):.1f}')
        wandb.log(step=engine.state.iteration, data={'valid_nll': valid_nll, 'valid_ppl': m.exp(valid_nll)})
        model_generate(model, sp_model)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Iteration {engine.state.iteration/grad_accum} - nll {engine.state.output:.3f} - lr {lr:.7f}')
        wandb.log(step=engine.state.iteration, data={'train_nll': engine.state.output})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
        logger.info(f'Finished epoch, current iteration {engine.state.iteration/grad_accum}')

    @trainer.on(Events.ITERATION_STARTED)
    def update_lr(engine):
        new_lr = lr_scheduler.step()
        optimizer.param_groups[0]['lr'] = new_lr

    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer, 'lr_scheduler': lr_scheduler}
    checkpointer = Checkpoint(to_save, params['workd'], n_saved=2, global_step_transform=global_step_from_engine(trainer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer)

    trainer.run(train_loader, max_epochs=params['epochs'])


def rms(x):
    return torch.mean(x**2)**0.5


def report_gradients(step, module, optimizer):
    found_grad = False
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is not None and param.ndim >= 2:
            found_grad = True
            # std = torch.std(param)
            norm = rms(param)
            # grad_std = torch.std(grad)
            grad_norm = rms(grad)
            exp_avg = optimizer.state[param]['exp_avg']
            grad_simil = F.cosine_similarity(grad.view(-1), exp_avg.view(-1), dim=0)
            data = {f"grad/{name}-rms": grad_norm.item(),
                f"param/{name}-rms": norm.item(),
                f"gradsim/{name}-cossim": grad_simil.item(),
                }
            wandb.log(data=data, step=step)
    if not found_grad:
        logger.info('Not found gradient.')


if __name__ == '__main__':
    main()
