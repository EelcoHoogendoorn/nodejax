"""Deferred configuration comparison in Flax NNX.

`Committee.__init__` consumes `members` and forwards four unused configuration arguments (`in_dim`, `width`, `depth`, `temperature`) down through `Tower.__init__` into `Block.__init__` (`THREADING_TAX = 5`), because NNX modules must execute concrete constructors eagerly before they can be composed. Retempering a trained model requires cloning the object graph and mutating `.temperature` across every `Block` instance in a nested loop.

Run directly with `python -m examples.comparisons.generics.generics_nnx`.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.comparisons.generics.generics_common import (
    CONFIGS, IN_DIM, LR, PARAM_KEY, RETEMPERED, TRAIN_STEPS, make_data, report)


class Block(nnx.Module):
    """Applies a same-width linear layer followed by temperature-scaled `tanh`.
    `temperature` sits in the graphdef rather than state, remaining static yet
    assignable on the mutable instance."""

    def __init__(self, width: int, temperature: float, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(width, width, rngs=rngs)
        self.temperature = temperature

    def __call__(self, rows: jax.Array) -> jax.Array:
        return jnp.tanh(self.linear(rows) / self.temperature)


class Tower(nnx.Module):
    """Chains `entry -> depth blocks -> readout`, forwarding `temperature` only
    so `Block` receives it."""

    def __init__(self, in_dim: int, width: int, depth: int,
                 temperature: float, *, rngs: nnx.Rngs):
        self.entry = nnx.Linear(in_dim, width, rngs=rngs)
        # nnx 0.12.8 refuses a plain list of submodules here ('use
        # nnx.List(...)', it says, among four other remedies): a
        # container of modules has to announce that it holds data
        self.blocks = nnx.List([Block(width, temperature, rngs=rngs)
                                for _ in range(depth)])
        self.readout = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, rows: jax.Array) -> jax.Array:
        carried = self.entry(rows)
        for block in self.blocks:
            carried = block(carried)
        return self.readout(carried)


class Committee(nnx.Module):
    """Averages `members` towers while forwarding four unused config parameters
    down to `Tower`."""

    def __init__(self, in_dim: int, width: int, depth: int, members: int,
                 temperature: float, *, rngs: nnx.Rngs):
        self.towers = nnx.List([Tower(in_dim, width, depth, temperature,
                                      rngs=rngs)
                                for _ in range(members)])

    def __call__(self, rows: jax.Array) -> jax.Array:
        return jnp.mean(jnp.stack([tower(rows) for tower in self.towers]), axis=0)


# Tower.temperature, and Committee's in_dim, width, depth, temperature
THREADING_TAX = 5


def configured(config: dict, seed: int) -> Committee:
    """Unpacks `config` into positional constructor arguments for Exercise 1,
    since every level restates what it forwards."""
    return Committee(IN_DIM, config['width'], config['depth'],
                     config['members'], config['temperature'],
                     rngs=nnx.Rngs(seed))


def fit(model: Committee, rows: jax.Array, targets: jax.Array) -> tuple:
    """Trains `model` directly with NNX graph-aware gradient and scan."""
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    carry_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(carry_axes, carry_axes, 0), out_axes=0)
    def train_step(model, optimizer, _):
        loss, grads = nnx.value_and_grad(
            lambda fitted: jnp.mean((fitted(rows) - targets) ** 2))(model)
        optimizer.update(model, grads)
        return loss

    losses = train_step(model, optimizer, jnp.arange(TRAIN_STEPS))
    return model, losses


def retempered(model: Committee, temperature: float) -> Committee:
    """Clones `model` and assigns `temperature` on every `Block` without
    rebuilding weights or keeping `config` at the call site.

    Two things it still costs. The walk is hand-written, so the caller
    must know the structure to reach every block, and it is a walk that
    grows with the architecture rather than one addressed edit. And the
    assignment mutates: the previous configuration is gone unless the
    model was cloned first, which is why this takes a clone."""
    flipped = nnx.clone(model)
    for tower in flipped.towers:
        for block in tower.blocks:
            block.temperature = temperature
    return flipped


def main() -> None:
    rows, targets = make_data()
    rows, targets = jnp.asarray(rows), jnp.asarray(targets)

    reported, first_trained = [], None
    for config in CONFIGS:
        model = configured(config, PARAM_KEY)
        model, losses = fit(model, rows, targets)
        parameters = sum(leaf.size for leaf in
                         jax.tree.leaves(nnx.state(model, nnx.Param)))
        reported.append((config, parameters, float(losses[0]), float(losses[-1])))
        first_trained = first_trained or model

    before = first_trained(rows)
    flipped = retempered(first_trained, RETEMPERED)
    shift = float(jnp.mean(jnp.abs(flipped(rows) - before)))

    # exercise 3: the model can be walked by hand, and nnx.graphdef holds
    # the static half, but neither is an address book: nothing here is a
    # spelling you could hand back to a builder to reconfigure by path.
    print('[flax nnx] the configuration, hand-walked off the model:')
    print(f'    members = {len(flipped.towers)}')
    print(f'    depth = {len(flipped.towers[0].blocks)}')
    print(f'    width = {flipped.towers[0].entry.out_features}')
    print(f'    temperature = {flipped.towers[0].blocks[0].temperature}')

    report('flax nnx', reported, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
