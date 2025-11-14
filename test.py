"""Test script for TensorSlow library."""

import numpy as np

from tensorslow.nn.loss import mse_loss
from tensorslow.tensor import Tensor

x = Tensor(np.linspace(-10, 10, 50))
y = x * 4 + 2 + Tensor(np.random.randn(50) * 3)

m = Tensor(np.random.randn())
b = Tensor(np.random.randn())
c = m * x + b
diff = mse_loss(c, y)
diff.backward()

print("a grad:", m.grad)
print("bias grad:", b.grad)
