import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pdb
import argparse
import os
import logging

from rnn_model import LSTM
from seq_preprocessing import SeqDataLoader, token_generation

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    print('Support CPU only')


def calc_accuracy(pred_scores, Y):
    with torch.no_grad():
        _, pred = torch.max(pred_scores, 1)
        train_acc = (pred == Y).float().mean()
        return train_acc.data.item()


ACCEPT = 'iclr/ICLR_accepted.xlsx'
REJECT = 'iclr/ICLR_rejected.xlsx'
MAX_LEN = 10

NUM_EPOCH = 35
BATCH_SIZE = 5

USE_CUDA = True
PRINT_EVERY = 20

EMBEDDING_DIM = 10
HIDDEN_DIM = 10
LR_DECAY_RATE = 1
OPTIMIZER = 'sgd_moment'
CLIP = 100

DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")

NAME = 'lstm_bs{}_hidden{}_embed{}_lrdc{}_clip{}_{}'.format(BATCH_SIZE, HIDDEN_DIM,
                                                  EMBEDDING_DIM, LR_DECAY_RATE, CLIP,
                                                  OPTIMIZER)
LOG_PATH = 'result/logs/lstm_/' + NAME
SAVE_PATH = 'result/ckpt/lstm_/' + NAME + '.pth'
CKPT_FILE = None
accepted = pd.read_excel(ACCEPT, index_col=0)
rejected = pd.read_excel(REJECT, index_col=0)

accepted.insert(1, "label", [1] * len(accepted))
rejected.insert(1, "label", [0] * len(rejected))

train_df = accepted[50:].append(rejected[50:])
test_df = accepted[:50].append(rejected[:50])

token_info = token_generation(accepted.append(rejected), True)

train_dl = SeqDataLoader(train_df, token_info, max_len=MAX_LEN, device=DEVICE)
test_dl = SeqDataLoader(test_df, token_info, max_len=MAX_LEN, device=DEVICE)
pad_idx = token_info['token_map']['<pad>']
lstm_model = LSTM(train_dl.n_token, EMBEDDING_DIM, HIDDEN_DIM, 2,
                  pad_idx).to(DEVICE)

writer_train = SummaryWriter(LOG_PATH + '/train')
writer_test = SummaryWriter(LOG_PATH + '/test')
loss_fn = nn.CrossEntropyLoss()

optims = {
    'adam': torch.optim.Adam(lstm_model.parameters()),
    'rmsprop': torch.optim.RMSprop(lstm_model.parameters()),
    'sgd': torch.optim.SGD(lstm_model.parameters(), 0.1),
    'sgd_moment': torch.optim.SGD(lstm_model.parameters(), 0.1, momentum=0.9)
}
optimizer = optims[OPTIMIZER]
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY_RATE)

if CKPT_FILE:
    print('Load checkpoint!!')
    checkpoint = torch.load(CKPT_FILE)
    lstm_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

step = 0
for epoch in range(NUM_EPOCH):

    lstm_model.train()
    train_iter = train_dl.batch_iter(bs=BATCH_SIZE)
    for seq, seq_len, labels in train_iter:
        lstm_model.zero_grad()
        pred_scores = lstm_model(seq, seq_len)
        loss = loss_fn(pred_scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(lstm_model.parameters(), CLIP)
        optimizer.step()

        if step % PRINT_EVERY == 0:
            writer_train.add_scalar('loss', loss.data.item(), step)
            writer_train.add_scalar('accuracy',
                                    calc_accuracy(pred_scores, labels), step)
            print('Train Loss = {}'.format(loss.data.item()))

            with torch.no_grad():
                acc_t, loss_list = [], []
                test_iter = test_dl.batch_iter(bs=len(test_dl))
                for seq_t, seq_len_t, labels_t in test_iter:
                    pred_scores_t = lstm_model(seq_t, seq_len_t)
                    loss_t = loss_fn(pred_scores_t, labels_t)
                    acc_t.append(calc_accuracy(pred_scores_t, labels_t))
                    loss_list.append(loss_t.data.item())
                writer_test.add_scalar('loss', np.mean(loss_list), step)
                writer_test.add_scalar('accuracy', np.mean(acc_t), step)
                print('Test Loss = {}'.format(np.mean(loss_list)))

        step += 1

    if epoch % 10 == 0:
        torch.save(
            {
                'model': lstm_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'BATCH_SIZE': BATCH_SIZE,
                'EPOCH': epoch,
                'STEP': step
            }, SAVE_PATH)
    scheduler.step()