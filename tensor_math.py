import torch
from torch.linalg import matrix_power

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# z1 = torch.empty(3)
# torch.add(x,y, out = z1)
# z2 = torch.add(x,y)
z = x+y
print(z)

#sub
z= x-y
print(z)
#div
z = torch.true_divide(x,y)
print(z)

#exponeation
z=x.pow(2)
z = x ** 2
print(z)

#simple comparison
z=x>0
z=x<0

#matrix multiplication
x1 = torch.rand((2,5))
print(x1)
x2 = torch.rand((5,3))
print(x2)
x3 = torch.mm(x1,x2)# 2*3
print(x3)
# x3 = x1.mm(x2)

#matrix exponenation

x1_exp = torch.rand((5,5))
print(x1_exp.matrix_power(3))

#element multi
print(x*y)

z = torch.dot(x,y)
print(z)

# batch matrix multilication
batch = 32
m=10
n=20
p=30

tensor1 = torch.rand((batch, n,m))
print(tensor1)
tensor = torch.rand((batch, m, p))
print(tensor)


