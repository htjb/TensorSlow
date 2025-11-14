"""Test script for TensorSlow library."""

import matplotlib.pyplot as plt
import numpy as np

from tensorslow.nn.loss import mse_loss
from tensorslow.tensor import Tensor

x = Tensor(np.linspace(-10, 10, 50))
y = x * 4 + 2 + Tensor(np.random.randn(50) * 3)

m = Tensor(np.random.randn())
b = Tensor(np.random.randn())

learning_rate = 0.001

loss = []
for i in range(1000):
    c = m * x + b
    loss = mse_loss(c, y)
    loss.backward()
    m.data -= learning_rate * m.grad
    b.data -= learning_rate * b.grad
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.data}")
    m.grad = np.zeros_like(m.data)
    b.grad = np.zeros_like(b.data)
    x.grad = np.zeros_like(x.data)
    y.grad = np.zeros_like(y.data)
    loss.grad = np.zeros_like(loss.data)

plt.plot(x.data, y.data, "o", label="Data")
plt.plot(x.data, (m.data * x.data + b.data), label="Fitted Line")
plt.legend()
plt.show()
