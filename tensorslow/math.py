"""Math operations for TensorSlow."""

import numpy as np

from tensorslow.tensor import Tensor
from tensorslow.utils import unreduce_grad


def exp(a: Tensor) -> Tensor:
    """Exponential function applied element-wise.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after applying exponential function.
    """
    out = Tensor(np.exp(a.data), (a,), "exp")

    def _backward() -> None:
        """Backward pass for exponential function."""
        # local derivative wrt a * out.grad
        a.grad += np.exp(a.data) * out.grad

    out._backward = _backward
    return out


def sum(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum of tensor elements along a specified axis.

    Args:
        a (Tensor): Input tensor.
        axis (int): Axis along which to sum.
            Defaults to None (sum over all axes).
        keepdims (bool, optional): Whether to keep the reduced dimensions.
            Defaults to False.

    Returns:
        Tensor: Output tensor after summation.
    """
    out = Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), (a,), "sum")

    def _backward() -> None:
        """Backward pass for summation."""
        a.grad += unreduce_grad(out.grad, a.shape, axis=axis)

    out._backward = _backward
    return out
