"""Activation checkpointing (`remat`) over `jax.checkpoint`."""

from __future__ import annotations

from typing import Callable

import jax

from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.transform import transform


@transform(preserves='param')
def remat(inner: Node, *, policy: Callable | None = None,
          prevent_cse: bool = True) -> Node:
    """Wraps `inner`'s `apply` call in `jax.checkpoint` to recompute activations during backprop.

    Placement in the composition tree controls checkpoint granularity: wrapping
    a single layer or cell checkpoints each step, whereas wrapping a `scan` or
    `stack` Node checkpoints the entire sequential pass.
    """
    def apply_fn(contract, param, state, input, rng):
        inner = contract.members.inner
        return jax.checkpoint(
            lambda child_rng, p, s, i: inner.apply(p, s, i, child_rng),
            policy=policy, prevent_cse=prevent_cse,
        )(rng, param, state, input)

    return Wrapper(inner=inner).roles(
        name=f'remat({inner.name})',
        apply=apply_fn,
    )
