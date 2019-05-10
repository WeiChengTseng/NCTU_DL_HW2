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

from rnn_model import RNN
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
NUM_EPOCH = 100
BATCH_SIZE = 50
USE_CUDA = True
PRINT_EVERY = 10
EMBEDDING_DIM = 10
HIDDEN_DIM = 10
DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
LOG_PATH = 'result/logs/rnn'

accepted = pd.read_excel(ACCEPT, index_col=0)
rejected = pd.read_excel(REJECT, index_col=0)

accepted.insert(1, "label", [1] * len(accepted))
rejected.insert(1, "label", [0] * len(rejected))

train_df = accepted[50:].append(rejected[50:])
test_df = accepted[:50].append(rejected[:50])

token_info = token_generation(accepted.append(rejected), True)

train_dl = SeqDataLoader(train_df, token_info, DEVICE)
test_dl = SeqDataLoader(test_df, token_info, DEVICE)

lstm_model = RNN(train_dl.n_token, EMBEDDING_DIM, HIDDEN_DIM, 2).to(DEVICE)

writer = SummaryWriter(LOG_PATH)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters())

step = 0
for epoch in range(NUM_EPOCH):

    lstm_model.train()
    train_iter = train_dl.batch_iter(bs=BATCH_SIZE)
    for seq, seq_len, labels in train_iter:
        lstm_model.zero_grad()
        pred_scores = lstm_model(seq, seq_len)
        loss = loss_fn(pred_scores, labels)
        loss.backward()
        optimizer.step()

        if step % PRINT_EVERY == 0:
            writer.add_scalar('train_loss', loss.data.item(), step)
            writer.add_scalar('train_acc', calc_accuracy(pred_scores, labels), step)
            print('Train Loss = {}'.format(loss.data.item()))
            
            with torch.no_grad():
                acc_t, loss_list = [], []
                test_iter = test_dl.batch_iter(bs=BATCH_SIZE)
                for seq_t, seq_len_t, labels_t in test_iter:
                    pred_scores_t = lstm_model(seq_t, seq_len_t)
                    loss_t = loss_fn(pred_scores_t, labels_t)
                    acc_t.append(calc_accuracy(pred_scores_t, labels_t))
                    loss_list.append(loss_t.data.item())
                writer.add_scalar('test_loss', np.mean(loss_list), step)
                writer.add_scalar('test_acc', np.mean(acc_t), step)
                print('Test Loss = {}'.format(np.mean(loss_list)))

        step += 1

writer.export_scalars_to_json(LOG_PATH + 'scalars.json')