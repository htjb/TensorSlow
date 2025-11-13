# TensorSlow


Building a minimal working ML library on top of numpy with automatic
differentiation.

## Example

The following script calculates the gradient over a sigmoid function with 
respect to two inputs a and b then plots a vs $\delta c/\delta a$.

```python
"""Test script for TensorSlow library."""

import matplotlib.pyplot as plt
import numpy as np

from tensorslow.nn.activations import sigmoid
from tensorslow.tensor import Tensor

a = Tensor(np.random.uniform(-10, 10, 50))
b = Tensor(1)

c = sigmoid(a * 2 + b)
c.backward()

print("a grad:", a.grad)
print("b grad:", b.grad)

plt.plot(a.data, a.grad, ".")
plt.title("Gradient of sigmoid at different inputs")
plt.xlabel("Input value")
plt.ylabel("Gradient value")
plt.grid()
plt.show()
```

## Style

The code base uses ruff to enforce google style docstrings, type hinting, 
formatting and other linting.