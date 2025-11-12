from tensorslow.tensor import Tensor
from tensorslow import math
import matplotlib.pyplot as plt
import numpy as np

a = Tensor(np.random.uniform(-10, 10, 50))
bias = Tensor(0.01)
weights = Tensor(np.random.uniform(-1, 1, (50, 5)).T)

c = math.sigmoid(math.sum(a*weights + bias, axis=1))
print("c data:", c.data)
c.backward()

print("a grad:", a.grad)
print("bias grad:", bias.grad)
print("weights grad:", weights.grad)