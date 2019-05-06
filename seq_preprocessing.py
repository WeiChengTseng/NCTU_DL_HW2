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
    def __init__(self, filename):
        seq_list = pd.read_excel(filename, index_col=0).values
        to_lower = lambda x: '<bos> ' + x[0].lower() + ' <eos>'
        self._seq = list(map(to_lower, seq_list))
        self._token = np.unique(' '.join(self._seq).split()) + ['<pad>']
        self._token_map = {word: idx for idx, word in enumerate(self._token)}
        self._idx_seq = None
        return

    def __len__(self):
        return len(self._seq)

    def batch_iter(self, bs=10):
        idx = 0
        self._shuffle()
        while (idx + bs < len(self)):
            src_len = np.array(
                [len(s[0]) for s in self._smiles_mapped[idx:idx + bs]],
                dtype=int)
            src_sort = np.argsort(src_len)[::-1]
            max_len = max(src_len)

            src = np.array([
                s[0] + [self._token_map['<pad>']] * (max_len - len(s[0]))
                for s in self._smiles_mapped[idx:idx + bs]
            ],
                           dtype=int)

            src_len, src = src_len[src_sort], src[src_sort]

            trg_len = np.array(
                [len(s[1]) for s in self._smiles_mapped[idx:idx + bs]],
                dtype=int)
            trg_sort = np.argsort(trg_len)[::-1]
            max_len = max(trg_len)
            trg = np.array([
                s[1] + [self._token_map['<pad>']] * (max_len - len(s[1]))
                for s in self._smiles_mapped[idx:idx + bs]
            ],
                           dtype=int)
            trg_len, trg = trg_len[trg_sort], trg[trg_sort]
            idx += bs

            if self._cuda:
                src_tensor = torch.Tensor(src)
                src_len_tensor = torch.Tensor(src_len)

                src_tensor = src_tensor.cuda()
                src_len_tensor = src_len_tensor.cuda()

                yield {
                    'src': src_tensor,
                    'src_len': src_len_tensor
                }
            else:
                yield {
                    'src': torch.Tensor(src),
                    'src_len': torch.Tensor(src_len)
                }

    def _shuffle(self):
        smiles_info = list(zip(self._smiles_mapped, self._smiles))
        random.shuffle(smiles_info)
        self._smiles_mapped, self._smiles = zip(*smiles_info)
        return


if __name__ == '__main__':
    dataset = SeqDataset('iclr/ICLR_accepted.xlsx')

        