"""Stochastic and regularizing layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def


@ambient
def Dropout(rate: float):
    """Dropout as a streaming stochastic node: rate is a STATIC (mode =
    which architecture you built; eval is the rate=0 build with the same
    params bound), the mask stream is rng STATE — a new mask every train
    step by auto-advance, no key threading."""
    def init(rng) -> Struct:
        return Struct(rng=rng)

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        if rate == 0.0:
            return state, input
        keep = jax.random.bernoulli(state.rng, 1.0 - rate, jnp.shape(input))
        return state, jnp.where(keep, input / (1.0 - rate), 0.0)

    return node_def(apply, init=init, name='drop')
