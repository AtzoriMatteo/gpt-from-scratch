# makemore

Character-level language models of increasing sophistication, trained to generate
plausible-looking new words (e.g. names) one character at a time.

**Reference:** Karpathy's "makemore" series (parts 1–3):
1. Bigram model
2. MLP (following Bengio et al. 2003)
3. Activations, gradients, and batchnorm — understanding what's actually happening
   inside training

## What's here

- `part1_bigram.py` — simplest possible model: predict the next character from just the current one
- `part2_mlp.py` — a proper multi-layer perceptron with an embedding lookup
- `part3_activations_batchnorm.py` — same MLP, but actually looking at activation
  statistics, gradient distributions, and adding batchnorm

## Data

Uses a plain text file of names, one per line (`../data/names.txt`) — not committed to
the repo (see `.gitignore`); download instructions in the script comments.

## What I learned

Part 1 (bigram): the counting model and the neural net really do converge to the same
thing. Counting gave an average NLL of 2.4544; the net, starting from random noise,
got to 2.4654 after 200 steps of plain gradient descent on a single 27x27 weight matrix.
Same ballpark, from two completely different starting points.

Writing softmax out by hand (exp, then divide by the row sum, instead of `F.softmax`)
made it click that "logits" are just log-counts — `W` isn't learning probabilities
directly, it's learning something that becomes probabilities after exponentiating and
normalizing. And the small L2 penalty on `W` during training is doing the exact same job
as the +1 smoothing in the count table: keep the model from betting everything on one
bigram it happened to see a lot.
