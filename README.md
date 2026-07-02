# gpt-from-scratch

A GPT-style language model built from scratch in PyTorch, following Andrej Karpathy's
["Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) series, implemented
and understood line by line rather than copy-pasted.

**Goal:** understand backpropagation, language modeling, and the transformer architecture
by building each piece myself, from a single scalar-valued autograd engine up to a working
GPT that trains and generates text.

## Structure

This repo follows the natural progression of the material:

```
gpt-from-scratch/
├── micrograd/     # a tiny scalar-valued autograd engine (the "why" behind backprop)
├── makemore/       # character-level language models, increasing in complexity
│   ├── part1_bigram.py
│   ├── part2_mlp.py
│   └── part3_activations_batchnorm.py
├── gpt/            # the full GPT: self-attention, transformer blocks, training loop
├── data/           # small datasets (gitignored if large)
└── requirements.txt
```

## Progress

- [ ] micrograd — scalar autograd engine
- [ ] makemore part 1 — bigram model
- [ ] makemore part 2 — MLP
- [ ] makemore part 3 — activations, gradients, batchnorm
- [ ] GPT — self-attention, multi-head attention, transformer blocks
- [ ] GPT — full training run + sample generation

## Notes

Each subfolder has its own short README with what that stage does and what I learned.
This is a learning project — the point is that I can explain every line, not that it's
production code.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
