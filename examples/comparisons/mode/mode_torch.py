"""The mode switch, the torch side of the comparison.

Self-contained: no nodejax import.

The famous spelling: ONE BIT ON THE OBJECT. model.train() and
model.eval() flip self.training on every submodule in place, and each
mode-dependent layer consults the bit at call time: Dropout draws or
passes through, BatchNorm pools the batch or reads its running stats.
The switch is one line, and its cost is discipline: the bit is global
mutable state on the model, every forward trusts whoever flipped it
last, and forgetting eval() before validation is the ecosystem's
best-known silent bug (the checks below would catch it: eval would
neither be deterministic nor isolated).

torch, like flax, is a reference-exhibit dependency: the file runs in
any environment with torch installed.

Run directly:  python -m examples.comparisons.mode.mode_torch
"""

import os

# torch and conda's mkl each ship a libomp on macOS; the flag lets one
# process host both copies so the import survives.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn

DIM, HIDDEN, CLASSES = 8, 16, 3
N, STEPS = 128, 200
RATE, MOMENTUM, LR = 0.3, 0.1, 0.02


def make_data(rs):
    centers = 2.0 * rs.standard_normal((CLASSES, DIM))
    labels = rs.randint(CLASSES, size=N)
    xs = centers[labels] + rs.standard_normal((N, DIM))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(labels)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, HIDDEN)
        self.drop = nn.Dropout(RATE)
        self.bn = nn.BatchNorm1d(HIDDEN, momentum=MOMENTUM)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x):
        return self.head(self.bn(self.drop(torch.nn.functional.gelu(self.lin(x)))))


def main() -> None:
    torch.manual_seed(0)
    xs, ys = make_data(np.random.RandomState(0))
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()                      # the bit, flipped on
    losses = []
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        logits_a = model(xs)           # still train mode: dropout draws,
        logits_b = model(xs)           # stats keep accumulating
        train_stochastic = not torch.allclose(logits_a, logits_b)

        # THE MODE SWITCH: one line, mutating the object in place
        model.eval()
        eval_a = model(xs)
        eval_b = model(xs)
        solo = model(xs[:16])

    print(f'{"torch":10s} train_stochastic={bool(train_stochastic)} '
          f'eval_deterministic={bool(torch.allclose(eval_a, eval_b))} '
          f'eval_isolated={bool(torch.allclose(solo, eval_a[:16], atol=1e-6))} '
          f'loss {losses[0]:.2f} -> {losses[-1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
