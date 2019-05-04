import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import tensorboardX

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.ImageFolder(root='./animal-10/train/',
                                           transform=data_transform)
