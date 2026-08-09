"""Recurrent neural network cell blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream


@ambient
def RNN(hidden: int):
    """Elman cell: h' = tanh(x Wx + h Wh + b), emitted as the output;
    state initializes to zeros shaped like the input."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            wx=0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            wh=0.4 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros(hidden))

    def init(ndef, param: Struct) -> jax.Array:
        return jnp.zeros_like(ndef.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input @ param.wx + state @ param.wh + param.b)
        return h, h

    return node_def(apply, init=init, param=param, name='rnn')
