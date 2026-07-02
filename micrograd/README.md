# micrograd

A tiny scalar-valued autograd engine — the smallest possible thing that does backpropagation.

**Why this comes first:** every neural network, no matter how large, is just this idea
(build a computation graph, then walk it backwards applying the chain rule) scaled up.
Building it at this scale makes backprop something you understand rather than something
you trust.

**Reference:** [Karpathy — "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)

## What's here

- `micrograd.py` — the `Value` class (scalar autograd) and a tiny MLP built on top of it

## What I learned

*(fill this in as you go — a few sentences on what clicked)*
