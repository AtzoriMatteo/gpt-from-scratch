"""
model.py -- a small GPT built from scratch.

Implement each piece yourself and understand it before moving to the next:
Head -> MultiHeadAttention -> FeedForward -> Block -> GPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        # TODO: key, query, value linear projections (no bias)
        # TODO: register a causal mask buffer (lower-triangular)
        raise NotImplementedError

    def forward(self, x):
        # TODO: compute attention scores (q @ k^T / sqrt(d)), apply causal mask,
        # softmax, then weighted sum of values
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        # TODO: a ModuleList of Head, plus an output projection
        raise NotImplementedError

    def forward(self, x):
        # TODO: run each head, concatenate, project
        raise NotImplementedError


class FeedForward(nn.Module):
    """A simple two-layer MLP applied per-position."""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # TODO: Linear -> ReLU -> Linear (expand to 4x then back down, per the paper)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    """Transformer block: communication (attention) then computation (feedforward)."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        # TODO: self-attention + feedforward, each with a residual connection
        # and a pre-LayerNorm
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        # TODO: token embedding table, position embedding table,
        # a stack of Blocks, a final LayerNorm, and an output linear head
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO: embed tokens + positions, run through blocks, project to
        # vocab logits; compute cross-entropy loss if targets given
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # TODO: autoregressive sampling loop -- crop context to block_size,
        # get logits for the last position, sample, append, repeat
        raise NotImplementedError
