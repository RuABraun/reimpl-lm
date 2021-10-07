import torch
from loguru import logger
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml

from data import TextData
from rnn import RnnLM


def main():

    params = load_hyperpyyaml(open('rnn.yaml'))

    train_data = TextData('train.pkl', params['seq_len'])
    valid_data = TextData('valid.pkl', 256)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                             num_workers=params['num_workers'])
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1,
                                             num_workers=1)

    model = RnnLM(100000, 256)

    criterion = nn.CrossEntropyLoss()

    model.cuda()
    optimizer = torch.optim.Adam(lr=params['lr'], params=model.parameters())
    logger.info('Started training.')
    for batch in train_loader:
        optimizer.zero_grad()
        inp_sent, targ_sent = batch
        pred = model(inp_sent)
        loss = criterion(targ_sent.view(-1), pred.view(-1, pred.size(2)))
        logger.info(f'Loss {loss}')
main()