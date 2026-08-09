"""Filtering and tracking layer blocks."""

from __future__ import annotations

import jax

from nodejax.ambient import ambient
from nodejax.authoring import node_def


@ambient
def EMA(tau: float):
    """One-pole low-pass over an arbitrary pytree: state is the smoothed
    tree, warm-started as a copy of the first offer. On a signal it is
    an ordinary smoothing filter; on a param pytree it is a target
    network."""
    def init(input):
        return input

    def apply(state, input):
        new = jax.tree.map(lambda s, i: tau * s + (1.0 - tau) * i, state, input)
        return new, new

    return node_def(apply, init=init, name='ema')
