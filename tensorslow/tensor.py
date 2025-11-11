from tensorslow.base import TensorBase
from tensorslow.utils import unbroadcast_grad

class Tensor(TensorBase):
    def __init__(self, data, _children=(), _op=""):
        super().__init__(data, _children, _op)
        Tensor.__add__ = add
        Tensor.__mul__ = mul
        Tensor.__sub__ = sub

def add(a, b):

    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data + b.data, (a, b), "+")

    def _backward():
        a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        b.grad += unbroadcast_grad(1 * out.grad, b.shape)
    out._backward = _backward
    return out

def mul(a, b):
    
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data * b.data, (a, b), '*')

    def _backward():
        # += local_derivative_wrt_a * out.grad
        a.grad += unbroadcast_grad(b.data * out.grad, a.shape)
        # += local_derivative_wrt_b * out.grad
        b.grad += unbroadcast_grad(a.data * out.grad, b.shape)
    out._backward = _backward
    return out

def sub(a, b):
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data - b.data, (a, b), "-")

    def _backward():
        a.grad += unbroadcast_grad(1 * out.grad, a.shape)
        b.grad += unbroadcast_grad(-1 * out.grad, b.shape)
    out._backward = _backward
    return out