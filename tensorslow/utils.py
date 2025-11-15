"""Utility functions for TensorSlow."""

import numpy as np

from tensorslow.base import TensorBase


def unreduce_grad(
    grad: np.ndarray,
    shape: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
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


def unbroadcast_grad(
    grad: np.ndarray, shape: int | tuple[int, ...]
) -> np.ndarray:
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


def zero_grad(a: TensorBase) -> None:
    """Reset gradients of all tensors in the computation graph.

    Args:
        a (Tensor): The output tensor from which to start the backpropagation.
    """
    topo, visited = [], set()

    def build_topo(v: TensorBase) -> None:
        """Build topological order of the computation graph.

        Traverse through the computation graph and for each tensor v
        explore the parents. Once all dependencies have been explored
        append v to the topology. Effectively checks each branch of
        the graph all the way back to initial tensors. topo has nodes
        ordered from input to ouput so needs reversing.
        """
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(a)
    for node in reversed(topo):
        node.grad = np.zeros_like(node.data, dtype=np.float64)
