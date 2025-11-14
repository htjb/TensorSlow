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
        requires_grad: bool = False,
    ) -> None:
        """Initialize a Tensor instance.

        Args:
            data (int | float | np.ndarray): The data for the tensor.
            _children (tuple, optional): Previous tensors in the
                computation graph. Defaults to ().
            _op (str, optional): Operation that produced this tensor.
                    Defaults to "".
            requires_grad (bool, optional): Whether to track gradients.
                Defaults to False.
        """
        super().__init__(data, _children, _op)
        self.requires_grad = requires_grad
        if not self.requires_grad:
            self.grad = np.zeros_like(self.data)


def add(a: Tensor, b: Tensor | int | float | np.ndarray) -> Tensor:
    """Element-wise addition of two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor | int | float | np.ndarray): Second input tensor or scalar.

    Returns:
        Tensor: Resultant tensor after addition.
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data + b.data, (a, b), "+", requires_grad=needs_grad)

    def _backward() -> None:
        """Backward pass for element-wise addition."""
        if a.requires_grad:
            a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        if b.requires_grad:
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
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data * b.data, (a, b), "*", requires_grad=needs_grad)

    def _backward() -> None:
        """Backward pass for element-wise multiplication."""
        # += local_derivative_wrt_a * out.grad
        if a.requires_grad:
            a.grad += unbroadcast_grad(b.data * out.grad, a.shape)
        # += local_derivative_wrt_b * out.grad
        if b.requires_grad:
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
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data - b.data, (a, b), "-", requires_grad=needs_grad)

    def _backward() -> None:
        """Backward pass for element-wise subtraction."""
        if a.requires_grad:
            a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast_grad(-1 * out.grad, b.shape)

    out._backward = _backward
    return out


def pow(a: Tensor, exponent: int | float) -> Tensor:
    """Element-wise power operation.

    Args:
        a (Tensor): Input tensor.
        exponent (int | float): The exponent to raise the tensor to.

    Returns:
        Tensor: Resultant tensor after exponentiation.
    """
    out = Tensor(
        a.data**exponent,
        (a,),
        f"pow_{exponent}",
        requires_grad=a.requires_grad,
    )

    def _backward() -> None:
        """Backward pass for power operation."""
        if a.requires_grad:
            a.grad += exponent * (a.data ** (exponent - 1)) * out.grad

    out._backward = _backward
    return out


def div(a: Tensor, b: Tensor | int | float | np.ndarray) -> Tensor:
    """Element-wise division of two tensors.

    Args:
        a (Tensor): Numerator tensor.
        b (Tensor | int | float | np.ndarray): Denominator tensor or scalar.

    Returns:
        Tensor: Resultant tensor after division.
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data / b.data, (a, b), "/", requires_grad=needs_grad)

    def _backward() -> None:
        """Backward pass for element-wise division."""
        if a.requires_grad:
            a.grad += unbroadcast_grad((1 / b.data) * out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast_grad(
                (-a.data / (b.data**2)) * out.grad, b.shape
            )

    out._backward = _backward
    return out

Tensor.__add__ = add
Tensor.__mul__ = mul
Tensor.__sub__ = sub
Tensor.__pow__ = pow
Tensor.__truediv__ = div
