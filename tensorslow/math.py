import numpy as np
from tensorslow.tensor import Tensor

def exp(a: Tensor):

    out = Tensor(np.exp(a.data), (a,), "exp")

    def _backward():
        # local derivative wrt a * out.grad
        a.grad += np.exp(a.data) * out.grad
    out._backward = _backward
    return out
