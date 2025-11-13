"""Standard Loss Functions for TensorSlow."""

from tensorslow.math import sum
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

    return out
