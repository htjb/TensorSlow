import numpy as np

class TensorBase():
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = np.zeros_like(data)
        self._prev = set(_children) # reference to previous tensors
        self._backward = lambda: None
        self._op = _op
    
    def backward(self):
        # all nodes in the graph making sure each only appears once...
        topo = []
        visited = set()

        def build_topo(v):
            """
            Traverse through the computation graph and for each tensor v
            explore the parents. Once all dependencies have been explored append
            v to the topology. Effectively checks each branch of the graph all
            the way back to initial tensors. topo has nodes ordered from input
            to ouput so needs reversing.
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            
        build_topo(self)
        # initialize the gradients (grad of tensor with respect to itself is 1)
        self.grad = np.ones_like(self.data)

        # iterate from ouput to input. Each nodes _backward() uses current .grad
        # accumulated gradient from downstream applies the chain rule and
        # updates grad of the parents.
        for node in reversed(topo):
            node._backward()