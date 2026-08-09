"""Embedding and Unembedding table blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream


@ambient
def Embed(vocab: int, hidden: int):
    """Token ids -> vectors by table lookup."""
    def param(rng: KeyStream) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.weight[input]

    return node_def(apply, param=param, name='embed')


@ambient
def Unembed(vocab: int, hidden: int):
    """Vectors -> vocab logits through a (vocab, hidden) matrix,
    transposed. Declares the same param structure as embed, so
    tie(pipe, 'embed', 'unembed') shares one matrix across both ends —
    the tied pipe never materializes this slot."""
    def param(rng: KeyStream) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.weight.T

    return node_def(apply, param=param, name='unembed')
