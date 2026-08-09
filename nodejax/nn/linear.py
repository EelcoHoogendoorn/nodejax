"""Linear / affine blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream


flat = node_def(lambda input: input.reshape(-1), name='flat')


@ambient
def Linear(n_out: int):
    """Affine map to n_out features; fan-in from the offer."""
    def param(ndef, rng: KeyStream) -> Struct:
        n_in = ndef.apply_input_spec.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return node_def(apply, param=param, name='linear')
