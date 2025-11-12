"""Tensor class and basic operations for TensorSlow."""

import numpy as np

from tensorslow.base import TensorBase
from tensorslow.utils import unbroadcast_grad


class Tensor(TensorBase):
    """Tensor class for automatic differentiation."""

    def __init__(
        self,
        data: int | float | np.ndarray,
        _children: tuple = (),
        _op: str = "",
    ) -> None:
        """Initialize a Tensor instance.

        Args:
            data (Union[int, float, np.ndarray]): The data for the tensor.
            _children (tuple, optional): Previous tensors in the
                computation graph. Defaults to ().
            _op (str, optional): Operation that produced this tensor.
                    Defaults to "".
        """
        super().__init__(data, _children, _op)
        Tensor.__add__ = add
        Tensor.__mul__ = mul
        Tensor.__sub__ = sub


def add(a: Tensor, b: Tensor | int | float | np.ndarray) -> Tensor:
    """Element-wise addition of two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor | int | float | np.ndarray): Second input tensor or scalar.

    Returns:
        Tensor: Resultant tensor after addition.
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data + b.data, (a, b), "+")

    def _backward() -> None:
        """Backward pass for element-wise addition."""
        a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        b.grad += unbroadcast_grad(1 * out.grad, b.shape)

    out._backward = _backward
    return out


def mul(a: Tensor, b: Tensor | int | float | np.ndarray) -> Tensor:
    """Element-wise multiplication of two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor | int | float | np.ndarray): Second input tensor or scalar.

    Returns:
        Tensor: Resultant tensor after multiplication.
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data * b.data, (a, b), "*")

    def _backward() -> None:
        """Backward pass for element-wise multiplication."""
        # += local_derivative_wrt_a * out.grad
        a.grad += unbroadcast_grad(b.data * out.grad, a.shape)
        # += local_derivative_wrt_b * out.grad
        b.grad += unbroadcast_grad(a.data * out.grad, b.shape)

    out._backward = _backward
    return out


def sub(a: Tensor, b: Tensor | int | float | np.ndarray) -> Tensor:
    """Element-wise subtraction of two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor | int | float | np.ndarray): Second input tensor or scalar.

    Returns:
        Tensor: Resultant tensor after subtraction.
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data - b.data, (a, b), "-")

    def _backward() -> None:
        """Backward pass for element-wise subtraction."""
        a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        b.grad += unbroadcast_grad(-1 * out.grad, b.shape)

    out._backward = _backward
    return out
