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
from seq_preprocessing import SeqDataLoader

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    print('Support CPU only')

ACCEPT = 'iclr/ICLR_accepted.xlsx'
REJECT = 'iclr/ICLR_rejected.xlsx'
NUM_EPOCH = 20
BATCH_SIZE = 50
USE_CUDA = True
PRINT_EVERY = 10
DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
LOG_PATH = 'result/logs/lstm'

accepted = pd.read_excel(ACCEPT, index_col=0)
rejected = pd.read_excel(REJECT, index_col=0)

accepted.insert(1, "label", [1] * len(accepted))
rejected.insert(1, "label", [0] * len(rejected))

train_df = accepted[50:].append(rejected[50:])
test_df = accepted[:50].append(rejected[:50])

train_dl = SeqDataLoader(train_df, DEVICE)
test_dl = SeqDataLoader(test_df, DEVICE)

lstm_model = LSTM(dataset.n_token, EMBEDDING_DIM, HIDDEN_DIM, dataset.n_token)

writer = SummaryWriter(LOG_PATH)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

step = 0
for epoch in range(NUM_EPOCH):

    lstm_model.train()
    train_iter = train_dl.batch_iter(bs=BATCH_SIZE, len_limit=80)
    for data in train_iter:
        lstm_model.zero_grad()

        sentence_in = data[args.domain]
        pad_tensor = torch.ones(BATCH_SIZE, 1)
        pad_tensor.new_full((BATCH_SIZE, 1), dataset.get_id('<pad>'))

        if args.gpu:
            pad_tensor = pad_tensor.cuda()
        targets = torch.cat([data[args.domain][:, 1:], pad_tensor], dim=1)

        pred_scores, _ = lstm_model(sentence_in.type(torch.LongTensor),
                               data[args.domain + '_len'])

        loss = loss_fn(pred_scores.permute(0, 2, 1),
                             targets.type(torch.LongTensor))
        loss.backward()
        optimizer.step()

        if step % PRINT_EVERY == 0:
            writer.add_scalar('train_loss', loss.data.item(), step)

        if step % PRINT_EVERY == 0:
            print('Loss = {}'.format(loss.data.item()))
        step += 1


writer.export_scalars_to_json(LOG_PATH + 'scalars.json')