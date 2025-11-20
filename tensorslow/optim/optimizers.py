"""Optimizer classes."""

from tensorslow.optim.optimizer_base import OptimizerBase


class GD(OptimizerBase):
    """Gradient Descent Optimizer."""

    def __init__(self, parameters: list, lr: float = 0.01) -> None:
        """Initialize the Gradient Descent optimizer.

        Args:
            parameters (list): List of tensors to optimize.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Perform a single optimization step."""
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad
