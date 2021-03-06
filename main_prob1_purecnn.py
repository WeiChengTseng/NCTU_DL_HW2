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
CKPT_FILE = None
# FC_BATCHNORM = True
# FC_DROPOUT = True
# MUL_FC = False
# K = 12
# DEPTH = 22
WEIGHT_DECAY = 1e-4
NAME = 'CNN_crop8_wd{}_dropout'.format(WEIGHT_DECAY)

NAME = 'SmallCNN5_crop8_wd{}_dropout'.format(WEIGHT_DECAY)
# NAME = 'ResNet_crop8_wd{}_dropout'.format(WEIGHT_DECAY)
LOG_PATH = 'result/logs/'+NAME
SVAE_PATH = 'result/ckpt/'+NAME+'.pth'

DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
print('use device: ', DEVICE)

writer_train = SummaryWriter(LOG_PATH + '/train')
writer_test = SummaryWriter(LOG_PATH + '/test')

data_transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop(224, scale=(0.8, 1.0)),
    # torchvision.transforms.RandomSizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

data_transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
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

model = SmallCNN().to(DEVICE)
# model = CNN().to(DEVICE)
# model = ResNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
# optimizer = optim.SGD(model.parameters(), lr=1e-1,
#                             momentum=0.9, weight_decay=WEIGHT_DECAY)
step = 0
print(NAME)
if CKPT_FILE:
    print('Load checkpoint!!')
    checkpoint = torch.load(CKPT_FILE)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
print(SVAE_PATH)
for epoch in range(NUM_EPOCH):

    for i, data in enumerate(train_dl):
        model.train()
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs.to(DEVICE))
        loss = criterion(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()

        if i % PRINT_EVERY == 0:
            optimizer.zero_grad()
            model.eval()
            print('[%d, %d] loss: %.3f' % (epoch, i, loss.item()))
            writer_train.add_scalar('loss', loss.item(), step)

            acc = calc_accuracy(outputs, labels.to(DEVICE))
            writer_train.add_scalar('accuracy', acc, step)

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
                writer_test.add_scalar('loss', np.mean(test_loss), step)
                writer_test.add_scalar('accuracy', np.mean(acc_test), step)
        step += 1
    if epoch % 10 == 0:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'K': K,
                # 'DEPTH': DEPTH,
                # 'FC_BATCHNORM': FC_BATCHNORM,
                # 'FC_DROPOUT': FC_DROPOUT,
                # 'MUL_FC': MUL_FC,
                'WEIGHT_DECAY': WEIGHT_DECAY
            },
            SVAE_PATH)

print('Finished Training')