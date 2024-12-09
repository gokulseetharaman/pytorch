import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.python.util.numpy_compat import np_array

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int, device=device)

print(my_tensor)
print(my_tensor.device)

#other init method

x = torch.empty(size = (3,3))
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
print(x)
x = torch.ones((3,3))
print(x)
x = torch.eye(5,5)
print(x)
x= torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1,5)).normal_(mean=0, std = 1)
print(x)


#how to iinit and convert tensors to other types
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())

#array to tensor

np_array = np.zeros((5,5))
np_ten = torch.from_numpy(np_array)


