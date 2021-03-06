import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pdb
import argparse
import os
import pandas as pd
import random
import pickle


class SeqDataLoader():
    def __init__(self, dataframe, token_info, max_len=10, device=None):
        df_value = dataframe.values
        seq_list, self._seq_label = df_value[:, 0], df_value[:, 1]
        to_lower = lambda x: '<bos> ' + x.lower() + ' <eos>'
        self._seq = list(map(to_lower, seq_list))
        self._token = list(np.unique(' '.join(self._seq).split())) + ['<pad>']

        # self._token_map = {word: idx for idx, word in enumerate(self._token)}
        # self._idx_map = {idx: word for idx, word in enumerate(self._token)}

        self._seq_max_len = max_len
        self._token_map = token_info['token_map']
        self._idx_map = token_info['idx_map']
        to_idx = lambda x: [self._token_map[token] for token in x.split()]
        self._idx_seq = list(map(to_idx, self._seq))
        self._device = device
        self.n_token = len(self._token_map)

        return

    def __len__(self):
        return len(self._seq)

    def batch_iter(self, bs=10):
        idx = 0
        self._shuffle()
        while (idx + bs <= len(self)):
            seq_len = np.array([
                min(len(s), self._seq_max_len)
                for s in self._idx_seq[idx:idx + bs]
            ],
                               dtype=int)
            seq_sort = np.argsort(seq_len)[::-1]
            seq_label = np.array(self._seq_label[idx:idx + bs])

            max_len = self._seq_max_len
            seq = np.array([
                s[:max_len] + [self._token_map['<pad>']] * max(
                    (max_len - len(s)), 0) for s in self._idx_seq[idx:idx + bs]
            ],
                           dtype=int)

            seq_len, seq, seq_label = seq_len[seq_sort], seq[
                seq_sort], seq_label[seq_sort]
            idx += bs

            yield torch.LongTensor(seq).to(
                self._device), torch.LongTensor(seq_len).to(
                    self._device), torch.LongTensor(seq_label).to(self._device)

    def _shuffle(self):
        seqs = list(zip(self._seq, self._idx_seq, self._seq_label))
        random.shuffle(seqs)
        self._seq, self._idx_seq, self._seq_label = zip(*seqs)
        return


def token_generation(dataframe, preload=False):
    if preload:
        return pickle.load(open('token_info.pkl', 'rb'))
    df_value = dataframe.values
    seq_list, seq_label = df_value[:, 0], df_value[:, 1]
    to_lower = lambda x: '<bos> ' + x.lower() + ' <eos>'
    seq = list(map(to_lower, seq_list))
    token = list(np.unique(' '.join(seq).split())) + ['<pad>']

    token_map = {word: idx for idx, word in enumerate(token)}
    idx_map = {idx: word for idx, word in enumerate(token)}
    pickle.dump({
        'token_map': token_map,
        'idx_map': idx_map
    }, open('token_info.pkl', 'wb'))
    return {'token_map': token_map, 'idx_map': idx_map}


if __name__ == '__main__':
    ACCEPT = 'iclr/ICLR_accepted.xlsx'
    REJECT = 'iclr/ICLR_rejected.xlsx'
    USE_CUDA = True
    DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                      and USE_CUDA) else torch.device("cpu")

    accepted = pd.read_excel(ACCEPT, index_col=0)
    rejected = pd.read_excel(REJECT, index_col=0)

    accepted.insert(1, "label", [1] * len(accepted))
    rejected.insert(1, "label", [0] * len(rejected))

    train_df = accepted[50:].append(rejected[50:])
    test_df = accepted[:50].append(rejected[:50])

    train_dl = SeqDataLoader(train_df, DEVICE)
