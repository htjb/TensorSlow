from tensorslow.tensor import Tensor
from tensorslow import math

a = Tensor(2.0)
b = Tensor(3.0)

c = a * b + a
c = math.exp(c)
c.backward()

print(c.data, type(c), c.grad)
print(a.grad, b.grad)