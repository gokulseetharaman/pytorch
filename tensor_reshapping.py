import torch

x  = torch.arange(9)

x_3x3 = x.view(3, 3) 
print(x_3x3)
x_3x3 = x.reshape(3, 3)
#both are same

y = x_3x3.t()#transpose
print(y)

print(y.reshape(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)#mo dimension transpose
print(z.shape)

x= torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

z = x.squeeze(0)
print(z.shape)

