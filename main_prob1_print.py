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

from cnn_model import CNN, DenseNet, SmallCNN, ResNet

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
except:
    print('Support CPU only')


def calc_accuracy(pred_scores, Y):
    with torch.no_grad():
        _, pred = torch.max(pred_scores, 1)
        train_acc = (pred == Y).float().mean()
        return train_acc.cpu().numpy()


NUM_EPOCH = 600
BATCH_SIZE = 50
USE_CUDA = True
PRINT_EVERY = 50
CKPT_FILE = 'result/ckpt/CNN_final.pth'
WEIGHT_DECAY = 1e-4


DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
print('use device: ', DEVICE)

data_transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

val_ds = torchvision.datasets.ImageFolder(root='./animal-10/val/',
                                          transform=data_transform_test)


val_dl = torch.utils.data.DataLoader(val_ds,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=4)

model = SmallCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

step = 0

if CKPT_FILE:
    print('Load checkpoint!!')
    checkpoint = torch.load(CKPT_FILE)
    model.load_state_dict(checkpoint['model'])


model.eval()
with torch.no_grad():
    test_loss, acc_test = [], []

    n_epoch_test = len(val_ds) // BATCH_SIZE
    for j, data_t in enumerate(val_dl):

        inputs_t, labels_t = data_t
        outputs_t = model(inputs_t.to(DEVICE))
        loss_t = criterion(outputs_t, labels_t.to(DEVICE))

        test_loss.append(loss_t.item())
        acc_test.append(
            calc_accuracy(outputs_t, labels_t.to(DEVICE)))
