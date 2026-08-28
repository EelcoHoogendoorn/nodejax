"""Linear / affine blocks."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import Node


flat = Leaf(lambda input: input.reshape(-1), name='flat')


def _fan_in_normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Normal weights scaled by the first, input-feature dimension."""
    return jax.random.normal(key, shape) / jnp.sqrt(shape[0])


@node
def Reshape(shape: tuple[int, ...]):
    """Reshape the signal to `shape`: geometry as a block, so a pipe can
    state its layout inline instead of minting a local lambda node."""
    return Leaf(lambda input: input.reshape(shape))


@node
def Linear(
    n_out: int,
    bias: bool = True,
    *,
    weight_init: Callable = _fan_in_normal,
    bias_init: Callable = jax.nn.initializers.zeros,
) -> Node:
    """Affine map to n_out features; fan-in from the resolved input
    spec. ``bias=False`` drops the offset: a pure linear map, for uses
    where an offset is meaningless (projecting a difference, feeding a
    normalization that would remove it). Initializers accept a key and
    shape; their defaults preserve the standard fan-in normal weights and
    zero bias."""
    def param(node, rng) -> Struct:
        n_in = node.input.shape[-1]
        weight = weight_init(rng.next(), (n_in, n_out))
        return (
            Struct(w=weight, b=bias_init(rng.next(), (n_out,)))
            if bias
            else Struct(w=weight)
        )

    def apply(param, input) -> jax.Array:
        product = input @ param.w
        return product + param.b if bias else product

    return Leaf(apply, param=param)


@node
def Projection(
    *,
    weight_init: Callable = _fan_in_normal,
    bias_init: Callable = jax.nn.initializers.zeros,
) -> Node:
    """Affine projection from one vector to one scalar.

    Initializers accept a key and shape; their defaults preserve the standard
    fan-in normal weights and zero bias.
    """
    def param(node, rng) -> Struct:
        width = node.input.shape[-1]
        return Struct(
            w=weight_init(rng.next(), (width,)),
            b=bias_init(rng.next(), ()),
        )

    def apply(param, input) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)
