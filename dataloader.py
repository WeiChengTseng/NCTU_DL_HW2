import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self._class = [
            'cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse',
            'squirrel', 'cow', 'elephant'
        ]
        self._files = []
        for i in self._class:
            
            pass
        return

    def __len__(self):
        return

    def __getitem__(self, idx):
        return