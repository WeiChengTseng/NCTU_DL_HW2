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

from cnn_model import CNN

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    print('Support CPU only')


def calc_accuracy(pred_scores, Y):
    with torch.no_grad():
        _, pred = torch.max(pred_scores, 1)
        train_acc = (pred == Y).float().mean()
        return train_acc.cpu().numpy()


NUM_EPOCH = 20
BATCH_SIZE = 50
USE_CUDA = True
PRINT_EVERY = 10
DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
print(DEVICE)

writer = SummaryWriter('result/logs/cnn')

data_transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

data_transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
train_ds = torchvision.datasets.ImageFolder(root='./animal-10/train/',
                                            transform=data_transform_train)
val_ds = torchvision.datasets.ImageFolder(root='./animal-10/val/',
                                          transform=data_transform_test)

train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=4)
val_dl = torch.utils.data.DataLoader(val_ds,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=4)

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
step = 0

for epoch in range(NUM_EPOCH):

    for i, data in enumerate(train_dl):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs.to(DEVICE))
        loss = criterion(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()

        if i % PRINT_EVERY == 0:
            print('[%d, %d] loss: %.3f' % (epoch, i, loss.item()))
            writer.add_scalar('train_loss', loss.item(), step)

            acc = calc_accuracy(outputs, labels.to(DEVICE))
            writer.add_scalar('train_acc', acc, step)

            with torch.no_grad():
                test_loss, acc_test = [], []
                optimizer.zero_grad()

                n_epoch_test = len(val_ds) // BATCH_SIZE
                for j, data_t in enumerate(val_dl):
                    inputs_t, labels_t = data_t
                    outputs_t = model(inputs_t.to(DEVICE))
                    loss_t = criterion(outputs_t, labels_t.to(DEVICE))

                    test_loss.append(loss_t.item())
                    acc_test.append(
                        calc_accuracy(outputs_t, labels_t.to(DEVICE)))
                writer.add_scalar('test_loss', np.mean(test_loss), step)
                writer.add_scalar('test_acc', np.mean(acc_test), step)
        step += 1

print('Finished Training')