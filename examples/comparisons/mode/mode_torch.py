"""Train and evaluation mode switching in PyTorch.

`model.train()` and `model.eval()` mutate `self.training` in place across all submodules. Every forward pass trusts whichever caller mutated the object most recently, and omitting `model.eval()` leaves dropout active and batch-norm buffers accumulating during evaluation.

Run directly with `python -m examples.comparisons.mode.mode_torch`.
"""

import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn

from examples.comparisons.mode.mode_common import (
    CLASSES, DIM, HIDDEN, LR, MOMENTUM, RATE, STEPS, make_data, report,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, HIDDEN)
        self.drop = nn.Dropout(RATE)
        self.bn = nn.BatchNorm1d(HIDDEN, momentum=MOMENTUM)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(self.drop(torch.nn.functional.gelu(self.lin(x)))))


def main() -> None:
    torch.manual_seed(0)
    xs_np, ys_np = make_data(np.random.RandomState(0))
    xs = torch.as_tensor(xs_np, dtype=torch.float32)
    ys = torch.as_tensor(ys_np, dtype=torch.long)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    losses = []
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        logits_a = model(xs)
        logits_b = model(xs)
        train_stochastic = not bool(torch.allclose(logits_a, logits_b))

        model.eval()
        eval_a = model(xs)
        eval_b = model(xs)
        solo = model(xs[:16])

    report(
        'torch',
        train_stochastic,
        bool(torch.allclose(eval_a, eval_b)),
        bool(torch.allclose(solo, eval_a[:16], atol=1e-6)),
        losses[0],
        losses[-1],
    )


if __name__ == '__main__':
    main()
