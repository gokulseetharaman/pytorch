import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customdataset import CatsAndDogsDataset

device = "cuda"

in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

#load data
dataset = CatsAndDogsDataset(csv_file='custom_dataset/cats_dogs.csv', root_dir='custom_dataset/cats_dogs_resized', transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True)

model = torchvision.models.googlenet(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features=1024, out_features=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        scores = model(data)

        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        print(f'cost at epoch {epoch} is {sum(losses) / len(losses)}')


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%')

    model.train()

print('checking accuracy on training set')
check_accuracy(train_loader, model)
print('checking accuracy on testing set')
check_accuracy(test_loader, model)



