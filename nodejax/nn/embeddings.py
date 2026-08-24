"""Embedding and Unembedding table blocks."""

from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.node import Node
from nodejax.ambient import node
from nodejax.authoring import Leaf


@node
def OneHot(classes: int) -> Node:
    """Integer labels to one-hot vectors on a new final axis."""
    return Leaf(lambda input: jax.nn.one_hot(input, classes))


@node
def Embed(vocab: int, hidden: int):
    """Token ids -> vectors by table lookup."""
    def param(rng) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param, input) -> jax.Array:
        return param.weight[input]

    return Leaf(apply, param=param)


@node
def Unembed(vocab: int, hidden: int):
    """Vectors -> vocab logits through a (vocab, hidden) matrix,
    transposed. Declares the same param structure as embed, so
    tie(pipe, 'embed', 'unembed') shares one matrix across both ends —
    the tied pipe never materializes this slot."""
    def param(rng) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param, input) -> jax.Array:
        return input @ param.weight.T

    return Leaf(apply, param=param)
