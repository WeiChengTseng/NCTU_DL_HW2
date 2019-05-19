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
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
except:
    print('Support CPU only')

def calc_accuracy(pred_scores, Y):
    try:
        with torch.no_grad():
            _, pred = torch.max(pred_scores, 1)
            train_acc = (pred == Y).float().mean()
            return train_acc.cpu().numpy()
    except:
            pred = np.argmax(pred_scores, 1)
            train_acc = (pred == Y).mean()
            return train_acc

NUM_EPOCH = 600
BATCH_SIZE = 20
USE_CUDA = True
PRINT_EVERY = 50
CKPT_FILE = 'result/ckpt/CNN_final.pth'
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda") if (torch.cuda.is_available()
                                  and USE_CUDA) else torch.device("cpu")
print('use device: ', DEVICE)

trans_ori = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

data_transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

val_ds = torchvision.datasets.ImageFolder(root='./animal-10/val/',
                                          transform=data_transform_test)
print_ds = torchvision.datasets.ImageFolder(root='./print/',
                                          transform=data_transform_test)
ori_ds = torchvision.datasets.ImageFolder(root='./print/',
                                          transform=trans_ori)

print(val_ds.classes)

val_dl = torch.utils.data.DataLoader(val_ds,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=4)
print_dl = torch.utils.data.DataLoader(print_ds,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=4)
ori_dl = torch.utils.data.DataLoader(ori_ds,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=4)

model = SmallCNN().to(DEVICE)

if CKPT_FILE:
    print('Load checkpoint!!')
    checkpoint = torch.load(CKPT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

model.eval()
with torch.no_grad():
    test_loss, acc_test = [], []

    n_epoch_test = len(print_ds) // BATCH_SIZE
    for j, data_t in enumerate(print_dl):

        inputs_t, labels_t = data_t
        outputs_t = model(inputs_t.to(DEVICE))
        _, pred = torch.max(outputs_t, 1)
        print(pred)
        print(labels_t)

output_all, label_all = [], []
with torch.no_grad():
    for data in val_dl:
        inputs, labels = data
        outputs = model(inputs.to(DEVICE))
        output_all.append(outputs)
        label_all.append(labels)
    output_all = torch.cat(output_all).cpu().numpy()
    label_all = torch.cat(label_all).cpu().numpy()
    _, pred_all = torch.max(output_all, 1)
    pred_all = pred_all.cpu().numpy()
    confusion_mat = confusion_matrix(label_all, pred_all)
    np.set_printoptions(precision=3)
    plot_confusion_matrix(label_all, pred_all, classes=val_ds.classes, normalize=True,
                      title='Confusion Matrix')
    plt.savefig('cm.png', dpi=400)
    for i in range(10):
        mask = label_all == i
        acc = calc_accuracy(output_all[mask], label_all[mask].cuda())
        print(acc)
        pass