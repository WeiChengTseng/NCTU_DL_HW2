import random
import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class AnimalDataset(Dataset):
    def __init__(self, path, transform=None):
        self._tsfm = transform
        self._class = [
            'cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse',
            'squirrel', 'cow', 'elephant'
        ]
        self._label_map = {c: idx for idx, c in enumerate(self._class)}
        self._files, self._labels = [], []
        for i in self._class:
            x = os.walk(os.path.join(path, i))
            class_path, _, filename = tuple(x)[0]
            self._files += [os.path.join(class_path, f) for f in filename]
            self._labels += [self._label_map[i]] * len(filename)
        return

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if self._tsfm:
            return self._tsfm(cv2.imread(self._files[idx],
                                         0)), self._labels[idx]
        else:
            return cv2.imread(self._files[idx], 0), self._labels[idx]


if __name__ == '__main__':
    dataset = AnimalDataset('./animal-10/train/')
    pdb.set_trace()