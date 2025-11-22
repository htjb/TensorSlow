"""Optimizer classes."""

import numpy as np

from tensorslow.optim.optimizer_base import OptimizerBase
from tensorslow.tensor import Tensor


class GD(OptimizerBase):
    """Gradient Descent Optimizer."""

    def __init__(self, parameters: list[Tensor], lr: float = 0.001) -> None:
        """Initialize the Gradient Descent optimizer.

        Args:
            parameters (list[Tensor]): List of tensors to optimize.
            lr (float, optional): Learning rate. Defaults to 0.001.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Perform a single optimization step."""
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad


class momentumGD(OptimizerBase):
    """Gradient descent with momentum."""

    def __init__(
        self, parameters: list[Tensor], lr: float = 0.001, alpha: float = 0.9
    ) -> None:
        """Initialize the momentum gradient descent class.

        Args:
            parameters (list[Tensor]): List of tensors to optimize.
            lr (float, optional): Learning rate. Defaults to 0.001.
            alpha (float, optional): Decay rate. Defaults to 0.9.
        """
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.updates = np.zeros_like(parameters)

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                parameter_update = (
                    -self.lr * param.grad + self.alpha * self.updates[i]
                )
                param.data += parameter_update
                self.updates[i] = parameter_update
