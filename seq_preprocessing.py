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

accecpted = pd.read_excel('iclr/ICLR_accepted.xlsx', index_col=0).values
rejected = pd.read_excel('iclr/ICLR_rejected.xlsx', index_col=0).values

class SeqDataset(Dataset):
    def __init__(self, filename):
        super(SeqDataset, self).__init__()
        seq_list = pd.read_excel(filename, index_col=0).values
        to_lower = lambda x: x[0].lower()
        self._seq = list(map(to_lower, seq_list))
        self._token = np.unique(' '.join(self._seq).split())
        print(self._token)
        return 


if __name__ == '__main__':
    dataset = SeqDataset('iclr/ICLR_accepted.xlsx')

        