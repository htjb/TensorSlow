
def add(a, b):
    from tensorslow.tensor import Tensor

    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data + b.data, (a, b), "+")


    def _backward():
        a.grad += 1 * out.grad
        b.grad += 1 * out.grad
    out._backward = _backward
    return out

def mul(a, b):
    from tensorslow.tensor import Tensor
    
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data * b.data, (a, b), '*')

    def _backward():
        # += local_derivative_wrt_a * out.grad
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    out._backward = _backward
    return out