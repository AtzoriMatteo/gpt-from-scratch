"""
makemore part 2 -- MLP language model (following Bengio et al. 2003).

Instead of a lookup table, embed characters into a continuous vector space and
feed a window of previous characters through a small neural network to predict
the next one.
"""

import torch
import torch.nn.functional as F


BLOCK_SIZE = 3  # how many previous characters we condition on


def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    # TODO: build (X, Y) tensors -- sliding windows of context -> next char
    raise NotImplementedError


class MLP:
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=200, block_size=BLOCK_SIZE):
        # TODO: initialize embedding table C, and weights/biases for the two layers
        raise NotImplementedError

    def forward(self, X):
        # TODO: embed -> concatenate context -> hidden layer (tanh) -> output logits
        raise NotImplementedError


if __name__ == "__main__":
    # TODO: load data (reuse part1's load_words), build dataset, train with
    # minibatch gradient descent, track loss with wandb
    pass
