"""Demo: trains a small MLP on a toy dataset using the micrograd engine."""
from micrograd import MLP
import random


# set seed for reproducibility
random.seed(42)

# initialise data
n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

# training loop
for k in range(41):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    for p in n.parameters():
        p.grad = 0.0
    # backward pass
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
