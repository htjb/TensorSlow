"""Test script for TensorSlow library."""

import matplotlib.pyplot as plt
import numpy as np

from tensorslow.nn.loss import mse_loss
from tensorslow.tensor import Tensor
from tensorslow.utils import zero_grad

x = Tensor(np.linspace(-10, 10, 500))
y = x * 4 + 2 + Tensor(np.random.randn(500) * 3)

m = Tensor(np.random.randn(), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

learning_rate = 0.0001

loss_history = []
for i in range(1000):
    c = m * x + b
    loss = mse_loss(c, y)
    loss.backward()
    m.data -= learning_rate * m.grad
    b.data -= learning_rate * b.grad
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.data}")
    zero_grad(loss)
    loss_history.append(loss.data)

plt.plot(x.data, y.data, "o", label="Data")
plt.plot(x.data, (m.data * x.data + b.data), label="Fitted Line")
plt.legend()
plt.show()

plt.plot(range(1000), loss_history, label="MSE Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()
