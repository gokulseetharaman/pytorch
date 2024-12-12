# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from fastai.layers import in_channels
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

#set device
device = "cuda"

#hyper parameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epoch = 10
load_model = True

#load data
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#init network
model = torchvision.models.vgg16(pretrained = True)
print(model)
sys.exit()

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_chekpoint(torch.load("my_checkpoint.pth.tar"))

#train
for epoch in range(num_epoch):
    losses = []

    if epoch %3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        #get data to cuda
        data = data.to(device=device)
        targets=targets.to(device=device)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        #backwards
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'loss of epoch {epoch} was {mean_loss:.5f}')

