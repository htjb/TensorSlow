import numpy as np


def unreduce_grad(grad, shape, axis=None):
    """
    Expand the gradients along reduced axes.
    """
    if axis is None:
        axis = tuple(range(len(shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    for ax in sorted(axis):
        grad = np.expand_dims(grad, ax)
    grad = np.broadcast_to(grad, shape)
    return grad

def unbroadcast_grad(grad, shape, axis=None):
    """
    Sum the gradients along axes that were broadcasted.
    """
    # collapse leading dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # sum along dimensions that were broadcasted
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad