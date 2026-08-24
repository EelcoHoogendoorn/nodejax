"""Linear / affine blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import node
from nodejax.authoring import Leaf


flat = Leaf(lambda input: input.reshape(-1), name='flat')


@node
def Reshape(shape: tuple[int, ...]):
    """Reshape the signal to `shape`: geometry as a block, so a pipe can
    state its layout inline instead of minting a local lambda node."""
    return Leaf(lambda input: input.reshape(shape))


@node
def Linear(n_out: int):
    """Affine map to n_out features; fan-in from the resolved input spec."""
    def param(node, rng) -> Struct:
        n_in = node.input.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param, input) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)
