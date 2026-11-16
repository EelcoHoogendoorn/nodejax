"""Defines stateless apply-time stochastic regularizers (`Dropout`, `DropPath`, `GaussianNoise`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.pnode import PNode
from nodejax.core.types import PyTree
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node(name='drop')
def Dropout(rate: float, train: bool = True) -> PNode:
    """Zeros independent elements with probability `rate` and rescales survivors by `1 / (1 - rate)` at `apply` time.

    Draws from `rng` during `apply` without holding state. Setting `train=False` builds an identity `Leaf` with no `rng` requirement while preserving an empty state tree so `specialize(**{'*.train': False})` rebinds existing state unchanged. Setting `rate=0.0` with `train=True` retains the stochastic `rng` contract.
    """
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng) -> jax.Array:
        keep = jax.random.bernoulli(rng.next(), 1.0 - rate, jnp.shape(input))
        return jnp.where(keep, input / (1.0 - rate), 0.0)

    return Leaf(apply)


@node(name='drop_path')
def DropPath(rate: float, train: bool = True) -> PNode:
    """Zeros an entire sample PyTree with probability `rate` and rescales kept samples by `1 / (1 - rate)` at `apply` time.

    Draws one scalar Bernoulli mask per call from `rng`, and builds an identity `Leaf` when `train=False`.
    """
    if rate < 0.0 or rate >= 1.0:
        raise ValueError('drop-path rate must satisfy 0 <= rate < 1')
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng) -> PyTree:
        keep = jax.random.bernoulli(rng.next(), 1.0 - rate)
        return jax.tree.map(
            lambda value: jnp.where(keep, value / (1.0 - rate), 0.0),
            input,
        )

    return Leaf(apply)


@node(name='gaussian_noise')
def GaussianNoise(std: float, train: bool = True) -> PNode:
    """Adds independent zero-mean Gaussian noise of standard deviation `std` to each leaf at `apply` time.

    Draws from `rng` during `apply`, and builds an identity `Leaf` when `train=False`.
    """
    if std < 0.0:
        raise ValueError('noise standard deviation must be nonnegative')
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng) -> PyTree:
        return jax.tree.map(
            lambda value: value + std * jax.random.normal(
                rng.next(), value.shape, dtype=value.dtype),
            input,
        )

    return Leaf(apply)
