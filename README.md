# gpt-from-scratch

A GPT-style language model built from scratch in PyTorch, following Andrej Karpathy's
["Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) series. Everything
here is implemented and understood line by line rather than copy-pasted, starting from a
single scalar autograd engine and working up to a transformer that actually trains and
generates text.

## What's here

- `micrograd/` — a tiny scalar-valued autograd engine, the "why" behind backprop
- `makemore/` — character-level language models, increasing in complexity
- `gpt/` — the full GPT: self-attention, transformer blocks, training loop
- `data/` — datasets, not committed (see `.gitignore`); each README has download instructions

Each subfolder has its own README with what that stage does and what I learned. This is a
learning project, so the point is being able to explain every line, not production code.

## Progress

- micrograd — done
- makemore part 1 (bigram) — done
- makemore part 2 (MLP) — not started
- makemore part 3 (activations, batchnorm) — not started
- GPT — not started

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
