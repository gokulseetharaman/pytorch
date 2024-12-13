from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.src.layers import Identity
from tensorflow.python.ops.gen_experimental_dataset_ops import load_dataset
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import sys


device = "cuda"

in_channels = 3
num_epoch = 5
learning_rate = 0.01
batch_size = 1024
num_channels = 10

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.vgg16(weights = None)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 100)
print(model)
sys.exit()
# for i in range(1, 7):
#     model.classifier[i] = Identity()

train_dataset = datasets.CIFAR10(root='dataset', train=True, transform=transforms.ToTensor, download=True)
train_loader = dataloader(datasets=train_dataset, batch_size=batch_size, shuffle = True)

criterian = nn.crossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

