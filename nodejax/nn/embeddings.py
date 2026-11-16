"""Defines one-hot encoding, token embedding, and vocabulary unembedding Leaf factories."""

from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def OneHot(classes: int) -> PNode:
    """Encodes integer class indices as one-hot vectors along a trailing axis of size `classes`."""
    return Leaf(lambda input: jax.nn.one_hot(input, classes))


@node
def Embed(vocab: int, hidden: int) -> Node:
    """Looks up `hidden`-dimensional vectors from a `(vocab, hidden)` table by integer token index."""
    def param(rng) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param, input) -> jax.Array:
        return param.weight[input]

    return Leaf(apply, param=param)


@node
def Unembed(vocab: int, hidden: int) -> Node:
    """Projects `hidden`-dimensional vectors to `vocab` logits through the transpose of a `(vocab, hidden)` weight matrix.

    Uses the same `Struct(weight=...)` parameter layout as `Embed(vocab, hidden)`, so `tie(pipe, 'embed', 'unembed')` shares one embedding matrix across both stages without allocating a second copy.
    """
    def param(rng) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param, input) -> jax.Array:
        return input @ param.weight.T

    return Leaf(apply, param=param)
