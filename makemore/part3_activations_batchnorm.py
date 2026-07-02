"""
makemore part 3 -- activations, gradients, and batchnorm.

Same MLP as part 2, but this time actually looking at what's happening inside
training: activation saturation, gradient magnitudes, and why batchnorm helps
stabilize deep networks.
"""

import torch


class MLPWithBatchNorm:
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=200, block_size=3):
        # TODO: same as part2's MLP, but add a BatchNorm1d-style layer after
        # the hidden linear layer, before the activation
        raise NotImplementedError

    def forward(self, X, training=True):
        # TODO: embed -> linear -> batchnorm -> tanh -> output logits
        # (batchnorm behaves differently at train vs. eval time -- that's the point)
        raise NotImplementedError


def plot_activation_stats(layer_outputs):
    # TODO: histogram of activations per layer -- are they saturating (stuck
    # near -1/1) or dead (stuck near 0)?
    raise NotImplementedError


def plot_gradient_stats(model):
    # TODO: histogram of gradients per parameter -- are they vanishing or
    # exploding anywhere?
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: train, then actually look at the activation/gradient plots --
    # this is the whole point of part 3, don't skip it
    pass
