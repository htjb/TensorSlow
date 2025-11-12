"""Common activation functions for neural networks."""

import numpy as np

from tensorslow.tensor import Tensor


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation function.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after applying sigmoid.
    """
    sig = 1 / (1 + np.exp(-a.data))
    out = Tensor(sig, (a,), "sigmoid")

    def _backward() -> None:
        """Backward pass for sigmoid activation."""
        a.grad += (sig * (1 - sig)) * out.grad

    out._backward = _backward
    return out
