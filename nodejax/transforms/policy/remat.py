"""Rematerialization: recompute the forward instead of storing it."""

from __future__ import annotations

from typing import Callable

import jax

from nodejax.core.node import Node
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def remat(inner: Node, *, policy: Callable | None = None,
          prevent_cse: bool = True) -> Node:
    """Checkpoint ``inner``, trading recomputation for lower backward memory.

    Placement controls granularity: wrapping a cell checkpoints each step;
    wrapping a scanned node checkpoints the complete rollout. ``policy`` and
    ``prevent_cse`` are passed to :func:`jax.checkpoint`.
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
