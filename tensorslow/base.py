"""Base Tensor class for TensorSlow library."""

import numpy as np


class TensorBase:
    """Base class for tensors in TensorSlow."""

    def __init__(
        self,
        data: int | float | np.ndarray,
        _children: tuple = (),
        _op: str = "",
    ) -> None:
        """Initialize a TensorBase instance.

        Args:
            data (Union[int, float, np.ndarray]): The data for the tensor.
            _children (tuple, optional): Previous tensors in the
                computation graph. Defaults to ().
            _op (str, optional): Operation that produced this tensor.
                    Defaults to "".

        Attributes:
            data (Union[int, float, np.ndarray]): The data of the tensor.
            grad (np.ndarray): The gradient of the tensor,
                initialized to zeros.
            _prev (set): Set of previous tensors in the computation graph.
            _backward (callable): Function to compute the backward pass.
            _op (str): Operation that produced this tensor.
            shape (tuple): Shape of the tensor data.

        """
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._prev = set(_children)  # reference to previous tensors
        self._backward = lambda: None
        self._op = _op

        if isinstance(data, int | float):
            self.shape = ()
        else:
            self.shape = self.data.shape

    def backward(self) -> None:
        """Perform backpropagation to compute gradients for the graph."""
        topo, visited = [], set()

        def build_topo(v: "TensorBase") -> None:
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

        build_topo(self)
        # initialize the gradients (grad of tensor with respect to itself is 1)
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # iterate from ouput to input. Each nodes _backward()
        # uses current .grad
        # accumulated gradient from downstream applies the chain rule and
        # updates grad of the parents.
        for node in reversed(topo):
            node._backward()
