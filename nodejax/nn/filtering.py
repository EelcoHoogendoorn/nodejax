"""Defines exponential moving-average smoothing over arbitrary PyTrees."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.pnode import PNode
from nodejax.core.types import PyTree


@node
def EMA(tau: float, warm: bool = False) -> PNode:
    """Tracks an exponential moving average `tau * state + (1 - tau) * input` over any PyTree.

    When `warm=False`, `init` allocates zeros matching `node.input` so initialization requires only an input spec. When `warm=True`, `init(input)` copies the first concrete input directly into state so smoothing starts at that sample rather than ramping up from zero, which requires a concrete input value at `init` time.
    """
    def init(input) -> PyTree:
        return input

    def init_cold(node) -> PyTree:
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input) -> tuple[PyTree, PyTree]:
        new = jax.tree.map(lambda s, i: tau * s + (1.0 - tau) * i, state, input)
        return new, new

    return Leaf(apply, init=init if warm else init_cold)
