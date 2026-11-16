"""Tied input/output embeddings in PyTorch.

Assigning `self.unembed.table = self.embed.table` shares one `nn.Parameter` instance across two submodules, and `model.parameters()` deduplicates parameters by object identity so `torch.optim` updates a single tensor with zero drift.

Run directly with `python -m examples.comparisons.tie.tie_torch`.
"""

import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn

from examples.comparisons.tie.tie_common import (
    DIM, LR, STEPS, VOCAB, make_data, report,
)


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.3 * torch.randn(VOCAB, DIM))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.table[token]


class RNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.wh = nn.Parameter(0.5 * torch.randn(DIM, DIM) / (DIM ** 0.5))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(x + self.wh @ h)
        return h, h


class Unembed(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.3 * torch.randn(VOCAB, DIM))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.table.T


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embed()
        self.cell = RNNCell()
        self.unembed = Unembed()
        self.unembed.table = self.embed.table

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(DIM)
        logits = []
        for token in ids:
            h, y = self.cell(h, self.embed(token))
            logits.append(self.unembed(y))
        return torch.stack(logits)


def main() -> None:
    torch.manual_seed(0)
    prev_np, cur_np = make_data(np.random.RandomState(0))
    prev = torch.as_tensor(prev_np, dtype=torch.long)
    cur = torch.as_tensor(cur_np, dtype=torch.long)
    model = LM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tables = [p for p in model.parameters() if p.shape == (VOCAB, DIM)]

    losses = []
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(prev), cur)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    drift = float((model.embed.table - model.unembed.table).abs().max().detach())
    report('torch', len(tables), drift, losses[0], losses[-1])


if __name__ == '__main__':
    main()
