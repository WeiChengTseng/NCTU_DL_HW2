import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import tensorboardX
import pdb
import math


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.bn2d3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 24 * 24, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.bn1d1 = nn.BatchNorm1d(256)
        self.bn1d2 = nn.BatchNorm1d(128)

        return

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool(F.relu(self.bn2d1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2d2(self.conv2(x))))
        x = self.pool(F.relu(self.bn2d3(self.conv3(x))))

        x = x.view(bs, -1)
        x = F.relu(F.dropout(self.bn1d1(self.fc1(x)), training=self.training))
        x = F.relu(F.dropout(self.bn1d2(self.fc2(x)), training=self.training))
        x = self.fc3(x)
        return x

class ExpCNN(nn.Module):
    def __init__(self, kernel, stride):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel, stride)
        self.conv2 = nn.Conv2d(64, 64, kernel, stride)

        self.pool = nn.MaxPool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.bn2d2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1d1 = nn.BatchNorm1d(256)

        return

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool(F.relu(self.bn2d1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2d2(self.conv2(x))))

        x = x.view(bs, -1)
        x = F.relu(F.dropout(self.bn1d1(self.fc1(x)), training=self.training))
        x = self.fc2(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.bn2d3 = nn.BatchNorm2d(64)
        self.bn2d4 = nn.BatchNorm2d(64)
        self.bn2d5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.bn1d1 = nn.BatchNorm1d(256)
        self.bn1d2 = nn.BatchNorm1d(128)

        return

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool(F.relu(self.bn2d1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2d2(self.conv2(x))))
        x = self.pool(F.relu(self.bn2d3(self.conv3(x))))
        x = self.pool(F.relu(self.bn2d4(self.conv4(x))))
        x = self.pool(F.relu(self.bn2d5(self.conv5(x))))

        x = x.view(bs, -1)
        x = F.relu(F.dropout(self.bn1d1(self.fc1(x)), training=self.training))
        x = F.relu(F.dropout(self.bn1d2(self.fc2(x)), training=self.training))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.pool = nn.MaxPool2d(3, 2)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.resblock1 = ResBlock(64)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2)
        self.resblock2 = ResBlock(64)

        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
        self.resblock3 = ResBlock(64)

        self.fc1 = nn.Linear(576, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.resblock1(x)
        x = self.conv2(x)
        x = self.pool2(self.resblock2(x))
        x = self.conv3(x)
        x = self.pool2(self.resblock3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(F.dropout(self.bn(self.fc1(x)), training=self.training))
        return self.fc2(x)


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        return

    def forward(self, x):
        inter = F.relu(self.bn1(self.conv1(x)))
        # print(inter.shape)
        out = self.bn2(self.conv2(inter))
        # print(out.shape)
        addition = out + x
        return F.relu(addition)


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               interChannels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels,
                               growthRate,
                               kernel_size=3,
                               padding=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               growthRate,
                               kernel_size=3,
                               padding=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               nOutChannels,
                               kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3,
                               nChannels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc1 = nn.Linear(4704, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.shape[0]
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 8)
        out = self.fc1(out.view(bs, -1))

        return out