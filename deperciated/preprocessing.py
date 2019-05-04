import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self._output_size = output_size

    def __call__(self, img, label):
        image, landmarks = img, label

        h, w = img.shape[:2]
        if isinstance(self._output_size, int):
            if h > w:
                new_h, new_w = self._output_size * h / w, self._output_size
            else:
                new_h, new_w = self._output_size, self._output_size * w / h
        else:
            new_h, new_w = self._output_size

        new_h, new_w = int(new_h), int(new_w)
        img_resize = transforms.resize(img, (new_h, new_w))


        landmarks = landmarks * [new_w / w, new_h / h]

        return img_resize, label


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self._output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self._output_size = output_size

    def __call__(self, img, label):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = img.shape[:2]
        new_h, new_w = self._output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = img[top: top + new_h,
                      left: left + new_w]

        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}