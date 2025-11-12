"""Utility functions for TensorSlow."""

import numpy as np


def unreduce_grad(
    grad: np.ndarray, shape: int | tuple, axis: int | tuple = None
) -> np.ndarray:
    """Expand the gradients along reduced axes.

    Args:
        grad (np.ndarray): The gradient to be expanded.
        shape (int | tuple): The target shape after unreducing.
        axis (int | tuple, optional): The axes that were reduced.
            If None, all axes are considered reduced. Defaults to None.

    Returns:
        np.ndarray: The unreduced gradient with the specified shape.
    """
    if axis is None:
        axis = tuple(range(len(shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    for ax in sorted(axis):
        grad = np.expand_dims(grad, ax)
    grad = np.broadcast_to(grad, shape)
    return grad


def unbroadcast_grad(grad: np.ndarray, shape: int | tuple) -> np.ndarray:
    """Adjust gradients for broadcasting.

    Args:
        grad (np.ndarray): The gradient to be adjusted.
        shape (int | tuple): The original shape of the tensor before
            broadcasting.

    Returns:
        np.ndarray: The adjusted gradient with the original shape.
    """
    # collapse leading dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # sum along dimensions that were broadcasted
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
