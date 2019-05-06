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

accecpted = pd.read_excel('iclr/ICLR_accepted.xlsx', index_col=0).values
rejected = pd.read_excel('iclr/ICLR_rejected.xlsx', index_col=0).values


class SeqDataset(Dataset):
    def __init__(self, filename):
        super(SeqDataset, self).__init__()
        seq_list = pd.read_excel(filename, index_col=0).values
        to_lower = lambda x: '<bos> ' + x[0].lower() + ' <eos>'
        self._seq = list(map(to_lower, seq_list))
        self._token = np.unique(' '.join(self._seq).split()) + ['<pad>']
        self._token_map = {word: idx for idx, word in enumerate(self._token)}
        return

    def __len__(self):

        return

    def __getitem__(self, idx):
        token_str = list(map(lambda x: self._token_map[x], self._seq[idx]))
        return token_str, len(token_str)


class SeqDataLoader():
    def __init__(self, filename, gpu=None):
        seq_list = pd.read_excel(filename, index_col=0).values
        to_lower = lambda x: '<bos> ' + x[0].lower() + ' <eos>'
        self._seq = list(map(to_lower, seq_list))
        self._token = np.unique(' '.join(self._seq).split()) + ['<pad>']
        self._token_map = {word: idx for idx, word in enumerate(self._token)}
        self._idx_map = {idx: word for idx, word in enumerate(self._token)}
        to_idx = lambda x: [self._token_map[token] for token in x]
        self._idx_seq = list(map(to_idx, self._seq))

        if gpu:
            self._cuda = True
        else:
            self._cuda = False
        return

    def __len__(self):
        return len(self._seq)

    def batch_iter(self, bs=10):
        idx = 0
        self._shuffle()
        while (idx + bs < len(self)):
            seq_len = np.array(
                [len(s[0]) for s in self._idx_seq[idx:idx + bs]], dtype=int)
            seq_sort = np.argsort(seq_len)[::-1]
            max_len = max(seq_len)

            seq = np.array([
                s[0] + [self._token_map['<pad>']] * (max_len - len(s[0]))
                for s in self._idx_seq[idx:idx + bs]
            ],
                           dtype=int)

            seq_len, seq = seq_len[seq_sort], seq[seq_sort]
            idx += bs

            if self._cuda:
                src_tensor = torch.Tensor(seq)
                src_len_tensor = torch.Tensor(seq_len)

                yield src_tensor.cuda(), src_len_tensor.cuda()
            else:
                yield torch.Tensor(seq), torch.Tensor(seq_len)

    def _shuffle(self):
        seqs = list(zip(self._seq, self._idx_seq))
        random.shuffle(seqs)
        self._seq, self._idx_seq = zip(*seqs)
        return


if __name__ == '__main__':
    dataset = SeqDataset('iclr/ICLR_accepted.xlsx')
