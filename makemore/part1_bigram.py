"""
makemore part 1 -- bigram language model.

Predicts the next character using only the current character (a lookup table of
counts/probabilities). The simplest possible language model: everything after
this is refining how the next character (given the context) is predicted.

Two implementations of the same model, as in the video:
  1. counting: a 27x27 count matrix, normalised into probabilities
  2. neural net: a single linear layer trained by gradient descent, which
     converges to more or less the same probabilities as counting

Data: a newline-separated list of words in ../data/names.txt
(e.g. https://raw.githubusercontent.com/karpathy/makemore/master/names.txt)
"""

import torch
import torch.nn.functional as F


def load_words(path="../data/names.txt"):
    """Reads names.txt and returns a list of names, one per line."""
    with open(path, "r") as f:
        return f.read().splitlines()


def build_vocab(words):
    """Creates the two lookup dictionaries: 'stoi', which maps char -> index, 
    and 'itos', which maps index -> char. A special start/end token '.' is used
    at index 0.
    """
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_bigram_counts(words, stoi):
    """Wrap each word as '.word.' and count, in a 27x27 matrix N, how many
    times character j follows character i. Row = current char, column = next.
    """
    vocab = len(stoi)
    N = torch.zeros((vocab, vocab), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N):
    """Turn counts into probabilities.

    Add 1 everywhere (smoothing, so no bigram is impossible and NLL can't
    hit infinity), then divide each row by its sum so row i becomes the 
    distribution "given char i, what comes next".
    """
    P = (N + 1).float()
    P /= P.sum(1, keepdim=True)
    return P


def sample(P, itos, g, n=5):
    """Generate names: start at '.' (index 0), draw the next character from
    the current character's row of P, repeat until '.' is drawn again.
    """
    names = []
    for _ in range(n):
        out = []
        ix = 0
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, replacement=True,
                                   generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


def avg_nll(P, words, stoi):
    """Measure model quality: average negative log likelihood of every
    bigram in the dataset. Lower = the model assigns higher probability to
    what actually occurs. This is the number the whole series keeps trying
    to push down.
    """
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return (-log_likelihood / n).item()


def build_training_set(words, stoi):
    """Reshape the same bigrams into supervised pairs: xs[i] is a character,
    ys[i] is the one that followed it. This is what the neural net trains on.
    """
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def train_nn(xs, ys, vocab, g, steps=200, lr=50.0):
    """The gradient-descent version of the bigram model.

    A vocab x vocab weight matrix W: forward pass (one-hot -> matmul
    -> softmax), cross-entropy loss plus a small L2 penalty (which
    plays the same role as the +1 smoothing in the counting model),
    backprop, update. Repeated for `steps` iterations.
    """
    W = torch.randn((vocab, vocab), generator=g, requires_grad=True)
    num = xs.nelement()
    xenc = F.one_hot(xs, num_classes=vocab).float()

    for k in range(steps):
        logits = xenc @ W  # log-counts
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num), ys].log().mean() \
               + 0.01 * (W ** 2).mean()

        W.grad = None
        loss.backward()

        W.data += -lr * W.grad

        if k % 20 == 0 or k == steps - 1:
            print(f"step {k:3d} | loss {loss.item():.4f}")

    return W


def nn_probs(W):
    """Convert the trained W into a probability table with the same shape as
    P, by exponentiating and row-normalizing.
    """
    counts = W.exp()
    return counts / counts.sum(1, keepdim=True)


if __name__ == "__main__":
    words = load_words()
    print(f"loaded {len(words)} words, e.g. {words[:5]}")

    stoi, itos = build_vocab(words)
    g = torch.Generator().manual_seed(2147483647)

    # --- counting model ---
    N = build_bigram_counts(words, stoi)
    P = counts_to_probs(N)
    print("\ncounting model samples:")
    for name in sample(P, itos, g):
        print(f"  {name}")
    print(f"counting model avg NLL: {avg_nll(P, words, stoi):.4f}")

    # --- neural net model ---
    print("\ntraining the neural net:")
    xs, ys = build_training_set(words, stoi)
    W = train_nn(xs, ys, vocab=len(stoi), g=g)
    Pnn = nn_probs(W)
    print("\nneural net samples:")
    for name in sample(Pnn, itos, g):
        print(f"  {name}")
    print(f"neural net avg NLL: {avg_nll(Pnn, words, stoi):.4f}")
