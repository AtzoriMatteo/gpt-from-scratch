"""
makemore part 1 -- bigram language model.

Predicts the next character using only the current character (a lookup table of
counts/probabilities). The simplest possible language model -- everything after
this is refining how we predict "next character given context."

Data: a newline-separated list of words in ../data/names.txt
(e.g. https://raw.githubusercontent.com/karpathy/makemore/master/names.txt)
"""

import torch


def load_words(path="../data/names.txt"):
    with open(path, "r") as f:
        return f.read().splitlines()


def build_vocab(words):
    # TODO: build char -> index and index -> char mappings, including a
    # special start/end token (commonly '.')
    raise NotImplementedError


def build_bigram_counts(words, stoi):
    # TODO: count how often each character follows each other character
    raise NotImplementedError


if __name__ == "__main__":
    words = load_words()
    print(f"loaded {len(words)} words, e.g. {words[:5]}")
    # TODO: build vocab, counts, normalize into probabilities, sample from the model
