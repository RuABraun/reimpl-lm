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
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss
from ignite.engine import Events, Engine

from data import TextDataIterable
from rnn import RnnLM


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


def model_generate(model, sp_model):
    with torch.no_grad():
        model.eval()
        sent = 'Rome was a great city'
        ids = sp_model.encode(sent)
        inp_sent = torch.tensor(ids, dtype=torch.int64).cuda()
        lst = sent.split()
        j = 0
        hidden = model.init_hidden(1)
        while j < 30:
            preds, _ = model(inp_sent.view(1, inp_sent.numel()), hidden)
            next_word_idx = F.gumbel_softmax(preds[:, -1], hard=True).argmax()
            next_word_idx_ = next_word_idx.cpu().item()
            lst.append(sp_model.id_to_piece(next_word_idx_))
            if next_word_idx_ == sp_model['.']:
                break
            inp_sent = torch.hstack((inp_sent, next_word_idx,))
            j += 1
        logger.info(f'Generated sentence: {lst}\n')


class CrossEntLoss(nn.CrossEntropyLoss):
    def __init__(self, weight = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input, target):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        return super().forward(input, target)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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
    params = load_hyperpyyaml(open('hparams/rnn.yaml'), overrides=args_dct)
    params['wd'] = float(params['wd'])
    params['lr'] = float(params['lr'])
    logger.info(f'Params are:\n{params}')

    wandb.init(project='nnlms', config=params, group='rnn')

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor('wikitext-103/50k_sp.model')
    vocab_size = len(sp_model)

    model = RnnLM(vocab_size, 256, params['dropout'])
    print(model)
    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f'Number of parameters: {num_params}')

    train_data = TextDataIterable('wikitext-103/train.pkl', params['seq_len'], params['batch_size'])
    valid_data = TextDataIterable('wikitext-103/valid.pkl', params['seq_len'], params['valid_batch_size'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'], drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params['valid_batch_size'],
                                               num_workers=0, drop_last=True)
    logger.info('Finished setting up dataloaders.')
    device = 'cuda'
    model.to(device)
    criterion = CrossEntLoss(reduction='mean')
    optimizer = torch.optim.AdamW(lr=params['lr'], 
            params=model.parameters(), weight_decay=params['wd'])
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = BasicLRScheduler(params['lr'], params['warmup_updates'])
    grad_accum = 1
    
    def update_step(engine, batch):
        optimizer.zero_grad()
        model.train()
        x, y = batch[0].cuda(), batch[1].cuda()
        with torch.cuda.amp.autocast():
            pred, h = model(x, engine.state.train_hidden)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        if engine.state.iteration % 20 == 0:
            report_gradients(engine.state.iteration, model, optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        engine.state.train_hidden = repackage_hidden(h)
        return loss
        
    trainer = Engine(update_step)

    def eval_step(engine, batch):
        with torch.no_grad():
            model.eval()
            x, y = batch[0].cuda(), batch[1].cuda()
            pred, h = model(x, engine.state.val_hidden)
            engine.state.val_hidden = h
        return pred, y

    evaluator = Engine(eval_step)

    loss_metric = Loss(criterion)
    loss_metric.attach(evaluator, "nll")

    @trainer.on(Events.EPOCH_STARTED)
    def init_hidden(engine):
        engine.state.train_hidden = model.init_hidden(params['batch_size'])

    @evaluator.on(Events.EPOCH_STARTED)
    def init_val_hidden(engine):
        engine.state.val_hidden = model.init_hidden(params['valid_batch_size'])

    @trainer.on(Events.ITERATION_COMPLETED(every=2000))
    def run_validation(engine):
        evaluator.run(valid_loader)
        valid_nll = evaluator.state.metrics['nll']
        logger.info(f'Iteration {engine.state.iteration} - valid nll {valid_nll:.3f} - ppl {m.exp(valid_nll):.1f}')
        wandb.log(step=engine.state.iteration, data={'valid_nll': valid_nll, 'valid_ppl': m.exp(valid_nll)})
        model_generate(model, sp_model)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Iteration {engine.state.iteration} - nll {engine.state.output*grad_accum:.3f} - lr {lr:.7f}')
        wandb.log(step=engine.state.iteration, data={'train_nll': engine.state.output*grad_accum})
        

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
