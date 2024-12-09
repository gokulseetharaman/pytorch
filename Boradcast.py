import torch
from sympy.physics.units.systems.si import dimex

x = torch.tensor([1,2,3])

#broadcasting
x1 = torch.rand(5,5)
x2 = torch.rand(1,5)

z= x1-x2
print(z)
z= x1**x2
print(z)

#other useful tensor opearation
sum_X = torch.sum(x, dim=0)
print(sum_X)
values, indeces = torch.max(x, dim=0)
values, indeces = torch.min(x, dim=0)

abs_x = torch.abs(x)
print(abs_x)
z=torch.argmax(x, dim=0)
z=torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)

#z = torch.eq(x,y)

sorted_y, indeces = torch.sort(x, dim=0, descending = False)
print(x)

z12 = torch.clamp(x, min=0)
print(z)

x1 = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z1 = torch.any(x1)
print(z1)
z1 = torch.all(x1)
print(z1)


