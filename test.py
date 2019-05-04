import torch
import torchvision

dataset = torchvision.datasets.ImageFolder('./animal-10/train/')

print(len(dataset))
for i in range(5):
    print(dataset[i+2000])