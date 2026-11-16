"""Defines affine projections and layout-reshaping Leaf nodes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import Node
from nodejax.core.pnode import PNode


flat = Leaf(lambda input: input.reshape(-1), name='flat')


def _fan_in_normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Draws normal weights scaled by the inverse square root of `shape[0]`."""
    return jax.random.normal(key, shape) / jnp.sqrt(shape[0])


@node
def Reshape(shape: tuple[int, ...]) -> PNode:
    """Reshapes an array input to `shape` as an inline pipeline stage."""
    return Leaf(lambda input: input.reshape(shape))


@node
def Linear(
    n_out: int,
    bias: bool = True,
    *,
    weight_init: Callable = _fan_in_normal,
    bias_init: Callable = jax.nn.initializers.zeros,
) -> Node:
    """Maps the trailing input axis to `n_out` features with fan-in inferred from `node.input.shape[-1]`.

    Setting `bias=False` omits `param.b` and applies a pure matrix product. `weight_init` and `bias_init` accept `(key, shape)` and default to fan-in-scaled normal weights and zeros.
    """
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
    """Projects a trailing feature vector to a scalar (`input @ param.w + param.b`).

    Infers the input width from `node.input.shape[-1]`. `weight_init` and `bias_init` accept `(key, shape)` and default to fan-in-scaled normal weights and zeros.
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
