

def unbroadcast_grad(grad, shape):
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