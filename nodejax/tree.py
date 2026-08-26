"""Generic operations over plain pytrees."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.types import PyTree


def tile(tree: PyTree, n: int) -> PyTree:
    """Tile a pytree along a new leading axis: the spelling for scanning
    a fixed batch n times."""
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n,) + jnp.shape(leaf)),
        tree,
    )
