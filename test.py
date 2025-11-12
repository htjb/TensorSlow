"""Test script for TensorSlow library."""

import matplotlib.pyplot as plt
import numpy as np

from tensorslow.nn.activations import sigmoid
from tensorslow.tensor import Tensor

a = Tensor(np.random.uniform(-10, 10, 50))
bias = Tensor(1)

c = sigmoid(a * 2 + bias)
c.backward()

print("a grad:", a.grad)
print("bias grad:", bias.grad)

plt.plot(a.data, a.grad, ".")
plt.title("Gradient of sigmoid at different inputs")
plt.xlabel("Input value")
plt.ylabel("Gradient value")
plt.grid()
plt.show()
