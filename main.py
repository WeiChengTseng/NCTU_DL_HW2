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

from cnn_model import CNN

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    print('Support CPU only')
    
NUM_EPOCH = 2
USE_CUDA = True
PRINT_EVERY = 10
DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")

writer = SummaryWriter('result/logs/')

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
train_ds = torchvision.datasets.ImageFolder(root='./animal-10/train/',
                                            transform=data_transform)
val_ds = torchvision.datasets.ImageFolder(root='./animal-10/val/',
                                          transform=data_transform)

train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=50,
                                       shuffle=True,
                                       num_workers=4)
val_dl = torch.utils.data.DataLoader(val_ds,
                                     batch_size=50,
                                     shuffle=True,
                                     num_workers=4)

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
step = 0

for epoch in range(NUM_EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_dl):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs.to(DEVICE))
        loss = criterion(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % PRINT_EVERY == 0:
            loss_ave = running_loss / PRINT_EVERY
            print('[%d, %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / PRINT_EVERY))
            writer.add_scalar('train_loss', loss_ave, step)
            running_loss = 0.0
        step += 1

print('Finished Training')