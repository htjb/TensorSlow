import numpy as np
#from tensorslow.utils import unbroadcast_grad
from tensorslow.tensor import Tensor

def exp(a: Tensor) -> Tensor:

    out = Tensor(np.exp(a.data), (a,), "exp")

    def _backward():
        # local derivative wrt a * out.grad
        a.grad += np.exp(a.data) * out.grad
    out._backward = _backward
    return out

def sigmoid(a: Tensor) -> Tensor:

    sig = 1 / (1 + np.exp(-a.data))
    out = Tensor(sig, (a,), "sigmoid")

    def _backward():
        # local derivative wrt a * out.grad
        a.grad += (sig * (1 - sig)) * out.grad
    out._backward = _backward
    return out

"""def sum(a, axis, keepdims=False):
    out = Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), (a,), "sum")

    def _backward():
        a.grad += unbroadcast_grad(out.grad, a.shape)
    out._backward = _backward
    return out"""