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


def tree_first(tree: PyTree) -> PyTree:
    """Take the first leading-axis element from every pytree leaf."""
    return jax.tree.map(lambda leaf: leaf[0], tree)


def tree_last(tree: PyTree) -> PyTree:
    """Take the last leading-axis element from every pytree leaf."""
    return jax.tree.map(lambda leaf: leaf[-1], tree)


def tree_stop_gradient(tree: PyTree) -> PyTree:
    """Preserve a pytree's values while stopping every leaf's gradient."""
    return jax.lax.stop_gradient(tree)
