"""Configuring a deep composition, the torch column.

The same three classes and the same explicit threading: Committee
consumes `members` and hands the other four values down, Tower carries
`temperature` only to reach Block, and `in_dim` is threaded from the
caller because nn.Linear must be told its fan-in.

Two differences from the jax columns, both named rather than hidden.
The training loop is an ordinary Python loop, because that is torch's
own idiom and it has no composable scan to write instead (what the
python loop costs in this setting is priced in the chunk comparison,
not here). And the flip after training is an assignment, as in nnx,
because a torch module is a mutable object.

Run directly:  python -m examples.comparisons.generics.generics_torch
"""

import numpy as np
import torch
from torch import nn

from examples.comparisons.generics.generics_common import (
    CONFIGS, IN_DIM, LR, PARAM_KEY, RETEMPERED, TRAIN_STEPS, make_data, report)


class Block(nn.Module):
    """The leaf. `temperature` is an ordinary Python attribute: torch
    keeps no register of what is configuration and what is data, so it
    is simply not a Parameter and nothing checks that."""

    def __init__(self, width: int, temperature: float):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.temperature = temperature

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(rows) / self.temperature)


class Tower(nn.Module):
    """entry -> depth blocks -> readout. `temperature` is here to reach
    Block and for nothing else."""

    def __init__(self, in_dim: int, width: int, depth: int, temperature: float):
        super().__init__()
        self.entry = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([Block(width, temperature)
                                     for _ in range(depth)])
        self.readout = nn.Linear(width, 1)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        carried = self.entry(rows)
        for block in self.blocks:
            carried = block(carried)
        return self.readout(carried)


class Committee(nn.Module):
    """`members` towers, averaged. Four of the five configuration
    parameters below are pure threading."""

    def __init__(self, in_dim: int, width: int, depth: int, members: int,
                 temperature: float):
        super().__init__()
        self.towers = nn.ModuleList([Tower(in_dim, width, depth, temperature)
                                     for _ in range(members)])

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.stack([tower(rows) for tower in self.towers]).mean(dim=0)


# Tower.temperature, and Committee's in_dim, width, depth, temperature
THREADING_TAX = 5


def configured(config: dict) -> Committee:
    """Exercise 1: the caller unpacks the dict and the constructor chain
    re-states every value at every level."""
    torch.manual_seed(PARAM_KEY)
    return Committee(IN_DIM, config['width'], config['depth'],
                     config['members'], config['temperature'])


def fit(model: Committee, rows: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Torch's own idiom: an eager Python loop over steps."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        loss = torch.mean((model(rows) - targets) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


def retempered(model: Committee, temperature: float) -> Committee:
    """Exercise 2: an assignment, as in nnx, and with the same two
    costs. The walk is hand-written and grows with the architecture,
    and the assignment mutates, so keeping the old configuration means
    copying the whole model first."""
    import copy

    flipped = copy.deepcopy(model)
    for tower in flipped.towers:
        for block in tower.blocks:
            block.temperature = temperature
    return flipped


def main() -> None:
    rows, targets = make_data()
    rows = torch.from_numpy(np.asarray(rows))
    targets = torch.from_numpy(np.asarray(targets))

    reported, first_trained = [], None
    for config in CONFIGS:
        model = configured(config)
        model, losses = fit(model, rows, targets)
        parameters = sum(int(weights.numel()) for weights in model.parameters())
        reported.append((config, parameters, losses[0], losses[-1]))
        first_trained = first_trained or model

    with torch.no_grad():
        before = first_trained(rows)
        flipped = retempered(first_trained, RETEMPERED)
        shift = float(torch.mean(torch.abs(flipped(rows) - before)))

    # exercise 3: named_modules() walks the tree and named_parameters()
    # walks the weights, but neither reports the CONFIGURATION: a plain
    # attribute is in no register, so what the model was built with
    # survives only in whatever dict the caller kept.
    print('[torch] the configuration, hand-walked off the model:')
    print(f'    members = {len(flipped.towers)}')
    print(f'    depth = {len(flipped.towers[0].blocks)}')
    print(f'    width = {flipped.towers[0].entry.out_features}')
    print(f'    temperature = {flipped.towers[0].blocks[0].temperature}')

    report('torch', reported, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
