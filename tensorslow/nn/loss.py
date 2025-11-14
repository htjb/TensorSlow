"""Standard Loss Functions for TensorSlow."""

import numpy as np

from tensorslow.math import abs, sum
from tensorslow.tensor import Tensor


def mse_loss(predicitons: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error Loss placeholder.

    Args:
        predicitons (Tensor): Predicted values.
        target (Tensor): Ground truth values.

    Returns:
        Tensor: Computed MSE loss.
    """
    out = sum((predicitons - target) ** 2) / predicitons.data.size

    def _backward() -> None:
        """Backward pass for MSE Loss."""
        predicitons.grad += (
            (2 / predicitons.data.size)
            * (predicitons.data - target.data)
            * out.grad
        )
        target.grad += (
            (-2 / predicitons.data.size)
            * (predicitons.data - target.data)
            * out.grad
        )
    out._backward = _backward
    return out


def mae_loss(predictions: Tensor, target: Tensor) -> Tensor:
    """Mean Absolute Error Loss placeholder.

    Args:
        predictions (Tensor): Predicted values.
        target (Tensor): Ground truth values.

    Returns:
        Tensor: Computed MAE loss.
    """
    out = sum(abs(predictions - target)) / predictions.data.size

    def _backward() -> None:
        """Backward pass for MAE Loss."""
        diff = predictions.data - target.data
        grad = (1 / predictions.data.size) * np.sign(diff) * out.grad
        predictions.grad += grad
        target.grad += -grad
    out._backward = _backward
    return out
