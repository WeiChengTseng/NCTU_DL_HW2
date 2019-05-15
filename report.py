import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pdb
import argparse
import os

def plot(trans_dl, trans_name, fig_name):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    ds_ori = torchvision.datasets.ImageFolder(root='./report/', transform=to_tensor)
    dl_ori = torch.utils.data.DataLoader(ds_ori,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=4)

    imgs = []

    for data in dl_ori:
        imgs.append(data[0].squeeze().permute(1,2,0).numpy())

    for i in range(3):
        for data in trans_dl:
            imgs.append(data[0].squeeze().permute(1,2,0).numpy())


    fig, axs = plt.subplots(1, len(imgs), constrained_layout=False, figsize=(8,3))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i])
        if i == 0:
            axs[i].set_title('original')
        axs[i].axis('off')
    fig.suptitle(trans_name, fontsize=18)
    plt.savefig(fig_name, dpi=400)
    return



random_sized_crop = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop(224, scale=(0.5, 1.0)),
    torchvision.transforms.ToTensor()
])
ds_trans = torchvision.datasets.ImageFolder(root='./report/', transform=random_sized_crop)
dl_trans = torch.utils.data.DataLoader(ds_trans,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=4)
plot(dl_trans, 'Random Resized Crop', 'rrc.png')

random_sized_crop = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
])
ds_trans = torchvision.datasets.ImageFolder(root='./report/', transform=random_sized_crop)
dl_trans = torch.utils.data.DataLoader(ds_trans,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=4)
plot(dl_trans, 'Random Flip', 'flip.png')