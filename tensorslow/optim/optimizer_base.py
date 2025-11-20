"""Optimizer base class."""

import numpy as np

from tensorslow.base import TensorBase


class OptimizerBase:
    """Base class for optimizers."""

    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    def zero_grad(self, a: TensorBase) -> None:
        """Reset gradients of all tensors in the computation graph.

        Args:
            a (TensorBase): The output tensor from which to
                start the backpropagation.
        """
        topo, visited = [], set()

        def build_topo(v: TensorBase) -> None:
            """Build topological order of the computation graph.

            Traverse through the computation graph and for each tensor v
            explore the parents. Once all dependencies have been explored
            append v to the topology. Effectively checks each branch of
            the graph all the way back to initial tensors. topo has nodes
            ordered from input to output so needs reversing.

            Args:
                v (TensorBase): Current tensor node being explored.
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(a)
        for node in reversed(topo):
            if hasattr(node, "requires_grad"):
                if node.requires_grad:
                    node.grad = np.zeros_like(node.data, dtype=np.float64)
