"""Tied embeddings, the torch side of the sharing comparison.

Self-contained: no nodejax import.

The model is built from components (an Embed block, an RNN cell, an
Unembed head), so the tie must cross a composition boundary: the two
views of the table live in different modules.

torch shares by OBJECT REFERENCE, and it is the standard practice the
ecosystem calls weight tying: assign one module's nn.Parameter into the
other after construction (the transformer idiom head.weight =
embedding.weight), and parameters() deduplicates by identity, so the
optimizer sees one table and both uses accumulate into its one .grad.
It works, and this file shows it holding: no drift, one copy.

As in flax, the sharing lives in Python object identity rather than in
any value the framework could serialize or transform: a state_dict
round trip writes the table twice under two keys, and whether a load
re-ties them is a per-tool convention, not a property of the model.

torch, like flax, is a reference-exhibit dependency: the file runs in
any environment with torch installed.

Run directly:  python -m nodejax.examples.comparisons.tie.tie_torch
"""

import os

# torch and conda's mkl each ship a libomp on macOS; the flag lets one
# process host both copies so the import survives.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn

VOCAB, DIM = 8, 6
POSITIONS, STEPS = 256, 200
LR = 0.03
CONCENTRATION = 2.0


def make_data(rs):
    logits = CONCENTRATION * rs.standard_normal((VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros(POSITIONS + 1, dtype=np.int64)
    state = rs.randint(VOCAB)
    for t in range(POSITIONS + 1):
        tokens[t] = state
        state = (P[state].cumsum() > rs.random()).argmax()
    return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.3 * torch.randn(VOCAB, DIM))

    def forward(self, token):
        return self.table[token]


class RNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.wh = nn.Parameter(0.5 * torch.randn(DIM, DIM) / DIM ** 0.5)

    def forward(self, h, x):
        h = torch.tanh(x + self.wh @ h)
        return h, h


class Unembed(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.3 * torch.randn(VOCAB, DIM))

    def forward(self, h):
        return h @ self.table.T


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embed()
        self.cell = RNNCell()
        self.unembed = Unembed()
        # the sharing crosses the component boundary after construction,
        # the transformer idiom head.weight = embedding.weight: one
        # Parameter object, two registrations, parameters() deduplicates
        self.unembed.table = self.embed.table

    def forward(self, ids):
        h = torch.zeros(DIM)
        logits = []
        for token in ids:
            h, y = self.cell(h, self.embed(token))
            logits.append(self.unembed(y))
        return torch.stack(logits)


def main() -> None:
    torch.manual_seed(0)
    prev, cur = make_data(np.random.RandomState(0))
    model = LM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tables = [p for p in model.parameters() if p.shape == (VOCAB, DIM)]

    losses = []
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(prev), cur)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    drift = float(
        (model.embed.table - model.unembed.table).abs().max().detach())
    print(f'{"torch":14s} table_copies={len(tables)} drift={drift:.2e} '
          f'loss {losses[0]:.2f} -> {losses[-1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
