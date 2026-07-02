# gpt

A small GPT-style transformer, trained on character-level text, built from the
ground up: self-attention, multi-head attention, transformer blocks, and the
training loop.

**Reference:** Karpathy — ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## What's here

- `model.py` — the GPT architecture (self-attention head, multi-head attention,
  feedforward, transformer block, full model)
- `train.py` — training loop, logging to W&B

## The goal

Every block below should be something I can explain out loud, not just code that
runs:
- [ ] single self-attention head
- [ ] multi-head attention
- [ ] positional embeddings
- [ ] transformer block (attention + feedforward + residual connections + layernorm)
- [ ] full model + training loop
- [ ] sample generation

## What I learned

*(fill this in as you go)*
