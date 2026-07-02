"""
train.py -- training loop for the from-scratch GPT, with W&B logging.
"""

import torch
# import wandb
# from model import GPT

# ---- hyperparameters ----
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path="../data/input.txt"):
    # TODO: load a plain text file (e.g. tiny shakespeare), build a
    # character-level vocab, encode the whole thing as a tensor of ints,
    # split into train/val
    raise NotImplementedError


def get_batch(data, block_size, batch_size, device):
    # TODO: sample random starting indices, build (x, y) batches where
    # y is x shifted by one character
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    # TODO: average loss over a few batches, for train and val, in eval mode
    raise NotImplementedError


def train():
    # wandb.init(project="gpt-from-scratch")
    # TODO: load data, build model, optimizer (AdamW), training loop with
    # periodic eval + logging, then generate a sample at the end
    raise NotImplementedError


if __name__ == "__main__":
    train()
