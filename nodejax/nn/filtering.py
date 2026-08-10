"""Filtering and tracking layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.ambient import ambient
from nodejax.authoring import node_def


@ambient
def EMA(tau: float, warm: bool = False):
    """One-pole low-pass over an arbitrary pytree: state is the smoothed
    tree. On a signal it is an ordinary smoothing filter; on a param
    pytree it is a target network.

    `warm` chooses where the smoothing STARTS, and with it what the node
    needs to be initialized. Warm copies the first real input, so the
    filter begins at the signal instead of decaying up to it from nothing
    — a target network starts equal to the online net — and the node then
    REQUIRES a real value at init. Cold starts at zeros, which needs only
    a shape, and pays a startup transient. A shape cannot stand in for a
    warm start: zeros are not the first sample."""
    def init(input):
        return input

    def init_cold(ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(state, input):
        new = jax.tree.map(lambda s, i: tau * s + (1.0 - tau) * i, state, input)
        return new, new

    return node_def(apply, init=init if warm else init_cold, name='ema')
