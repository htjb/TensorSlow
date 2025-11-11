from tensorslow.tensor import Tensor
from tensorslow import math
import numpy as np

a = Tensor(np.random.uniform(0, 1, size=(5, 2)))
b = Tensor(3.0)

c = b + b * a
c = math.exp(c)
c.backward()

print(c.data, type(c), c.grad)
print(a.grad, b.grad)