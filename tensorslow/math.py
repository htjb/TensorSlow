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
    out = Tensor(np.exp(a.data), (a,), "exp", requires_grad=a.requires_grad)

    def _backward() -> None:
        """Backward pass for exponential function."""
        # local derivative wrt a * out.grad
        if a.requires_grad:
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
        axis (int | tuple[int, ...] | None): Axis along which to sum.
            Defaults to None (sum over all axes).
        keepdims (bool, optional): Whether to keep the reduced dimensions.
            Defaults to False.

    Returns:
        Tensor: Output tensor after summation.
    """
    out = Tensor(
        np.sum(a.data, axis=axis, keepdims=keepdims),
        (a,),
        "sum",
        requires_grad=a.requires_grad,
    )

    def _backward() -> None:
        """Backward pass for summation."""
        if a.requires_grad:
            a.grad += unreduce_grad(out.grad, a.shape, axis=axis)

    out._backward = _backward
    return out


def abs(a: Tensor) -> Tensor:
    """Absolute value function applied element-wise.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after applying absolute value function.
    """
    out = Tensor(np.abs(a.data), (a,), "abs", requires_grad=a.requires_grad)

    def _backward() -> None:
        """Backward pass for absolute value function."""
        if a.requires_grad:
            a.grad += np.sign(a.data) * out.grad

    out._backward = _backward
    return out
