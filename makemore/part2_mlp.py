"""
makemore part 2 -- MLP language model (following Bengio et al. 2003).
 
Instead of a lookup table, embed characters into a continuous vector space and
feed a window of previous characters through a small neural network to predict
the next one.
"""
 
import random
import torch
import torch.nn.functional as F
import wandb
from part1_bigram import load_words, build_vocab
 
BLOCK_SIZE = 3  # denotes how many previous characters we condition on


def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    """Build (X, Y) tensors, sliding windows of context -> next char.
 
    X has shape (n_examples, block_size), Y has shape (n_examples,), both
    int tensors. Each word is padded with '.' on both ends: the context
    starts as all-'.' and slides forward one character at a time, so the
    model sees examples like ('...' -> 'e'), ('..e' -> 'm'), ('.em' -> 'm')
    for the word "emma".
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    return torch.tensor(X), torch.tensor(Y)


class MLP:
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=200, block_size=BLOCK_SIZE,
                 g=None):
        """Initialise the embedding table C and the weights/biases for the
        two layers.
 
        C: (vocab_size, embed_dim): one row per character, what gets learned 
           put into an embedding space.
        W1: (block_size * embed_dim, hidden_dim), b1: (hidden_dim,),
           the hidden layer; input dim is block_size * embed_dim because
           the context's embeddings get concatenated before this layer.
        W2: (hidden_dim, vocab_size), b2: (vocab_size,),
           projects the hidden state to logits over the vocabulary.
        """
        self.block_size = block_size
        self.C = torch.randn((vocab_size, embed_dim), generator=g)
        self.W1 = torch.randn((block_size * embed_dim, hidden_dim), generator=g)
        self.b1 = torch.randn(hidden_dim, generator=g)
        self.W2 = torch.randn((hidden_dim, vocab_size), generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)
        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.params:
            p.requires_grad = True
 
    def forward(self, X):
        """Embed -> concatenate context -> hidden layer (tanh) -> output logits.
 
        X: (B, block_size) int tensor of context indices.
        Returns logits of shape (B, vocab_size).
        """
        emb = self.C[X]  # (B, block_size, embed_dim)
        h = torch.tanh(emb.view(emb.shape[0], -1) @ self.W1 + self.b1)  # (B, hidden_dim)
        logits = h @ self.W2 + self.b2  # (B, vocab_size)
        return logits
 
    def __call__(self, X):
        return self.forward(X)


def build_splits(words, stoi, block_size=BLOCK_SIZE, seed=42):
    """Shuffle the words and split 80/10/10 into train/dev/test.
 
    Dev is for tuning and checking overfitting; test gets touched only once 
    at the very end.
    """
    words = words[:]
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return (build_dataset(words[:n1], stoi, block_size),
            build_dataset(words[n1:n2], stoi, block_size),
            build_dataset(words[n2:], stoi, block_size))
 
 
@torch.no_grad()
def split_loss(model, name, X, Y):
    """Evaluate loss on a whole split (train/dev/test) with gradients off."""
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    print(f"{name:5s} loss: {loss.item():.4f}")
    return loss.item()
 
 
@torch.no_grad()
def sample(model, itos, g, n=10):
    """Generate names by repeatedly predicting the next character from the
    rolling context, stopping when '.' is drawn.
    """
    names = []
    for _ in range(n):
        out = []
        context = [0] * model.block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names
 
 
if __name__ == "__main__":
    BATCH_SIZE = 32
    STEPS = 200_000
 
    wandb.init(project="makemore", name="part2-mlp", config={
        "block_size": BLOCK_SIZE,
        "embed_dim": 10,
        "hidden_dim": 200,
        "batch_size": BATCH_SIZE,
        "steps": STEPS,
    })
 
    words = load_words()
    stoi, itos = build_vocab(words)
 
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = build_splits(words, stoi)
    print(f"train {Xtr.shape[0]} | dev {Xdev.shape[0]} | test {Xte.shape[0]}")
 
    g = torch.Generator().manual_seed(2147483647)
    model = MLP(vocab_size=len(stoi), g=g)
    print(f"{sum(p.nelement() for p in model.params)} parameters")
 
    for i in range(STEPS):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
 
        # forward -- F.cross_entropy fuses softmax + NLL and is numerically
        # stable (subtracts the max logit internally, so no exp() overflow)
        logits = model(Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix])
 
        # backward
        for p in model.params:
            p.grad = None
        loss.backward()
 
        # update with step decay
        lr = 0.1 if i < 100_000 else 0.01
        for p in model.params:
            p.data += -lr * p.grad
 
        if i % 1_000 == 0:
            wandb.log({"train/batch_loss": loss.item(), "lr": lr}, step=i)
        if i % 10_000 == 0:
            print(f"step {i:6d} | batch loss {loss.item():.4f}")
 
    train_loss = split_loss(model, "train", Xtr, Ytr)
    dev_loss = split_loss(model, "dev", Xdev, Ydev)
    wandb.log({"final/train_loss": train_loss, "final/dev_loss": dev_loss})
    # uncomment only when you're completely done tuning:
    # split_loss(model, "test", Xte, Yte)
 
    print("\nsamples:")
    g_sample = torch.Generator().manual_seed(2147483647 + 10)
    for name in sample(model, itos, g_sample):
        print(f"  {name}")
 
    wandb.finish()
