"""
micrograd — a tiny scalar-valued autograd engine.

Following Karpathy's "spelled-out intro to neural networks and backpropagation."
Build this yourself, line by line — don't just copy it in. The goal is that by
the end you could rebuild this from a blank file without the video.
"""


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        # internal bookkeeping for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        # TODO: implement forward pass + local backward closure
        raise NotImplementedError

    def __mul__(self, other):
        # TODO: implement forward pass + local backward closure
        raise NotImplementedError

    def tanh(self):
        # TODO: implement the activation + its derivative
        raise NotImplementedError

    def backward(self):
        # TODO: topological sort, then walk the graph backwards
        # calling each node's _backward()
        raise NotImplementedError

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    # sanity check once implemented:
    # a = Value(2.0); b = Value(-3.0); c = a * b; c.backward()
    # print(a.grad, b.grad)
    pass
