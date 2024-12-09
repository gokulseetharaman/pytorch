import torch

batch_size = 10
feautures = 25
x = torch.rand(batch_size, feautures)
print(x[0].shape)

print(x[:, 0].shape)

print(x[2, 0:10])

x[0,0] = 100
print(x[0])

x=torch.arange(10)
indeces = [2, 5, 8]
print(x[indeces])
print(x)

x = torch.rand((3,5))
print(x)
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])

print(x[rows, cols])

#advanced indexing
x =torch.arange(10)
print(x[(x<2) & (x>8)])
print(x[x.remainder(2) == 0])

#useful operation
print(torch.where(x>5, x, x*2))
print(torch.tensor([0,0,1,1,2,2,3,4,5,6,6,7,8]).unique())
print(x.ndimension())
print(x.numel())

