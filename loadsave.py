# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from fastai.layers import in_channels
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # default stride=(1,1), padding=(1,1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print('saving checkpoint')
    torch.save(state, filename)

def load_chekpoint(checkpoint):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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
model = CNN(in_channel=in_channels, num_classes=num_classes).to(device)

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

