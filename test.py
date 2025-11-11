from tensorslow.tensor import Tensor
from tensorslow import math
import matplotlib.pyplot as plt
import numpy as np

a = Tensor(np.random.uniform(-10, 10, 50))
bias = Tensor(3.0)
weights = Tensor(np.random.uniform(-1, 1, (50, 5)).T)

c = math.sigmoid(a*weights + bias)
print("c data:", c.data)
c.backward()

plt.plot(a.data, c.data, 'o', label='sigmoid(a)')
plt.plot(a.data, a.grad, 'o', label='grad of sigmoid(a)')
plt.show()