"""Stochastic and regularizing layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node(name='drop')
def Dropout(rate: float, train: bool = True) -> Node:
    """Dropout drawing at APPLY: the key arrives beside the input (the
    pipe splits its one boundary key toward every drawing member), one
    draw per call, no state anywhere.

    MODE IS A BUILD, and `train` is the static that says which: at
    train=False the factory returns the identity. BOTH builds are
    stateless, so the train/eval flip never touches the state tree,
    which is what lets a state-bound model specialize across it
    verbatim. The BOOLEAN is the only kind toggle: a rate of zero stays
    a full stochastic build that happens to keep everything, because a
    numerical value changing the signature is a trap."""
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng) -> jax.Array:
        keep = jax.random.bernoulli(rng.next(), 1.0 - rate, jnp.shape(input))
        return jnp.where(keep, input / (1.0 - rate), 0.0)

    return Leaf(apply)


@node(name='drop_path')
def DropPath(rate: float, train: bool = True) -> Node:
    """Drop an entire residual branch with one draw per example."""
    if rate < 0.0 or rate >= 1.0:
        raise ValueError('drop-path rate must satisfy 0 <= rate < 1')
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng):
        keep = jax.random.bernoulli(rng.next(), 1.0 - rate)
        return jax.tree.map(
            lambda value: jnp.where(keep, value / (1.0 - rate), 0.0),
            input,
        )

    return Leaf(apply)


@node(name='gaussian_noise')
def GaussianNoise(std: float, train: bool = True) -> Node:
    """Add independent Gaussian noise to every array leaf at apply time."""
    if std < 0.0:
        raise ValueError('noise standard deviation must be nonnegative')
    if not train:
        return Leaf(lambda input: input)

    def apply(input, rng):
        return jax.tree.map(
            lambda value: value + std * jax.random.normal(
                rng.next(), value.shape, dtype=value.dtype),
            input,
        )

    return Leaf(apply)
