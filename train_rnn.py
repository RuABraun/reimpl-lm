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
from torchinfo import summary

from data import TextDataIterable, load_vocab, SB
from rnn import RnnLM


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def main():

    logger.info('Starting script.')
    import sys
    args = sys.argv[1:]
    args_dct = {arg.split('=')[0]: arg.split('=')[1] for arg in args}
    
    params = load_hyperpyyaml(open('rnn.yaml'), overrides=args_dct)
    
    symtab = load_vocab('wikitext-103/vocab')

    params['wd'] = float(params['wd'])
    wandb.init(project='rnnlm', config=params)

    cross_batch_context = params['cross_batch_context']

    vocab_size = len(symtab)
    model = RnnLM(vocab_size, 256, use_resblock=params['use_resblock'], dropout=params['dropout'])
    summary(model, input_size=(4, 16,), dtypes=[torch.LongTensor], device='cpu')

    train_data = TextDataIterable('wikitext-103/train.pkl', params['seq_len'], params['batch_size'] if cross_batch_context else None)
    valid_data = TextDataIterable('wikitext-103/valid.pkl', 256, 4)

    logger.info(f'One epoch will take {len(train_data) // (params["seq_len"]*params["batch_size"])} iterations')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'], drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=2,
                                               num_workers=0)
    logger.info('Finished setting up dataloader.')
    criterion = nn.CrossEntropyLoss()

    model.cuda()
    
    optimizer = torch.optim.Adam(lr=params['lr'], params=model.parameters(), weight_decay=params['wd'])
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr=params['lr'] * 0.1, verbose=True)
    logger.info(f'Started training with params:\n{params}')
    iternum = 0
    avg_loss = None
    model_params = model.parameters()
    for epoch in range(params['epochs']):
        logger.info(f'Starting epoch {epoch+1}')
        hidden = model.init_hidden(params["batch_size"])
        for i, batch in enumerate(train_loader):
            iternum += 1
            if params['warmup'] and iternum < 1001:
                optimizer.param_groups[0]['lr'] = (iternum / 1000) * params['lr']
            optimizer.zero_grad()
            inp_sent, targ_sent = batch
            inp_sent = inp_sent.cuda()
            targ_sent = targ_sent.cuda()

            pred, hidden = model(inp_sent, hidden)

            loss = criterion(pred, targ_sent.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model_params, max_norm=params['clip_norm'])
            optimizer.step()

            hidden = repackage_hidden(hidden)

            if avg_loss is None:
                avg_loss = loss.item()
            else:
                avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
            if iternum % 200 == 0:
                logger.info(f'Iter {iternum} - loss {loss.item():.2f} - avg loss {avg_loss:.2f}')
                report_gradients(iternum, model)

            if iternum % 500 == 0:
                with torch.no_grad():
                    model.eval()
                    valid_loss = 0.
                    valid_batch_cnt = 0
                    for batch in valid_loader:
                        valid_batch_cnt += 1
                        inp_sent, targ_sent = batch
                        inp_sent = inp_sent.cuda()
                        targ_sent = targ_sent.cuda()
                        pred, _ = model(inp_sent)
                        loss = criterion(pred, targ_sent.view(-1))
                        valid_loss += loss.item()
                    valid_loss /= valid_batch_cnt
                    logger.info(f'Valid loss {valid_loss:.2f} ppl {m.exp(valid_loss)}')
                    lr_scheduler.step(valid_loss)
                model.train()
                wandb.log({'train_avg_loss': avg_loss, 'valid_loss': valid_loss})

            if iternum % 1000 == 0:
                with torch.no_grad():
                    model.eval()
                    inp_sent = torch.tensor(symtab[SB], dtype=torch.int64).cuda()
                    lst = [SB]
                    for j in range(20):
                        preds, _ = model(inp_sent.view(1, j+1))
                        next_word_idx = F.gumbel_softmax(preds[-1], hard=True).argmax()
                        lst.append(symtab[next_word_idx.cpu().item()])
                        inp_sent = torch.hstack((inp_sent, next_word_idx,))
                    logger.info(f'Generated sentence: {lst}\n')
                model.train()

            if iternum == params['max_updates']:
                torch.save(model.state_dict(), params['workd'] + '/model.pt')
                import sys; sys.exit(0)

    torch.save(model.state_dict(), params['workd'] + '/model.pt')


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
