import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AnimalDataset(Dataset):
    def __init__(self, path):
        self._class = [
            'cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse',
            'squirrel', 'cow', 'elephant'
        ]
        self._files = []
        for i in self._class:
            # root, dirs, files = os.walk(os.path.join(path,i))
            x = os.walk(os.path.join(path,i))
            print(x)
            # for f in files:
            #     print(os.path.join(root, f))

            break
        return

    def __len__(self):
        return

    def __getitem__(self, idx):
        return

if __name__ == '__main__':
    dataset = AnimalDataset('./animal-10/train/')