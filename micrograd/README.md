# micrograd

A tiny scalar-valued autograd engine, the smallest possible thing that does backpropagation.

Every neural network, no matter how large, is just this idea
(build a computation graph, then walk it backwards applying the chain rule) scaled up.
Building it at this scale helps understand backprop starting from zero.

Reference: [Karpathy — "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)

## What's here

- 'micrograd.py' — the Value class (scalar autograd) plus Neuron, Layer and MLP built on top of it.
- 'train.py' — trains a small MLP (3 → 4 → 4 → 1) on a toy dataset with squared error and plain gradient descent.

To run the training demo:

```bash
python train.py
```

## What I learned

Backpropagation turned out to be a lot more mathematics than it looks like: behind
every operation there's a local derivative and a chain rule step, but at the same
time it's far less code than I expected. The entire engine fits in one small file,
and most of it is just each operation knowing how to pass gradients back to its inputs.

What impressed me most is how much such a tiny architecture can already do: an MLP
built out of scalar operations, trained with nothing but gradient descent, fits the
toy dataset in just a few dozen steps. Everything bigger (PyTorch, GPTs) is this 
exact mechanism, just vectorised and made bigger.
