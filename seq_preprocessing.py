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


class SeqDataLoader():
    def __init__(self, dataframe, device=None):
        # seq_list = pd.read_excel(filename, index_col=0).values
        seq_list = dataframe.values
        to_lower = lambda x: '<bos> ' + x[0].lower() + ' <eos>'
        self._seq = list(map(to_lower, seq_list))
        self._token = list(np.unique(' '.join(self._seq).split())) + ['<pad>']
        # print(self._token)
        # pdb.set_trace()
        self._token_map = {word: idx for idx, word in enumerate(self._token)}
        self._idx_map = {idx: word for idx, word in enumerate(self._token)}
        to_idx = lambda x: [self._token_map[token] for token in x.split()]
        self._idx_seq = list(map(to_idx, self._seq))
        self._device = device

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

            yield torch.Tensor(seq).to(self._device), torch.Tensor(seq_len).to(
                self._device)

    def _shuffle(self):
        seqs = list(zip(self._seq, self._idx_seq))
        random.shuffle(seqs)
        self._seq, self._idx_seq = zip(*seqs)
        return


if __name__ == '__main__':
    dataset = SeqDataLoader('iclr/ICLR_accepted.xlsx')
