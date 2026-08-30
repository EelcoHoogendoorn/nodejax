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


def tree_tail(tree: PyTree) -> PyTree:
    """Drop the first leading-axis element from every leaf."""
    return jax.tree.map(lambda leaf: leaf[1:], tree)


def tree_len(tree: PyTree) -> int:
    """Return the common leading-axis length of a non-empty pytree."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError('tree_len needs at least one leaf')
    shapes = tuple(jnp.shape(leaf) for leaf in leaves)
    if any(not shape for shape in shapes):
        raise ValueError('tree_len needs every leaf to have a leading axis')
    length = shapes[0][0]
    if any(shape[0] != length for shape in shapes[1:]):
        raise ValueError('tree_len found unequal leading-axis lengths')
    return length


def tree_reshape(tree: PyTree, shape: tuple, axes: int = 1) -> PyTree:
    """Replace the first ``axes`` leading axes of every leaf with ``shape``."""
    return jax.tree.map(
        lambda leaf: leaf.reshape(shape + jnp.shape(leaf)[axes:]),
        tree,
    )


def tree_swap_axes(tree: PyTree, axis_a: int, axis_b: int) -> PyTree:
    """Swap two axes of every leaf."""
    return jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, axis_a, axis_b),
        tree,
    )


def tree_take(tree: PyTree, indices) -> PyTree:
    """Gather rows of every leaf's leading axis by an index array."""
    return jax.tree.map(lambda leaf: leaf[indices], tree)


def tree_broadcast_axis(tree: PyTree, count: int, axis: int) -> PyTree:
    """Broadcast a pytree over one new axis."""
    def broadcast(leaf):
        shape = jnp.shape(leaf)
        normalized_axis = axis if axis >= 0 else len(shape) + axis + 1
        expanded = jnp.expand_dims(leaf, normalized_axis)
        target = shape[:normalized_axis] + (count,) + shape[normalized_axis:]
        return jnp.broadcast_to(expanded, target)

    return jax.tree.map(broadcast, tree)


def tree_stop_gradient(tree: PyTree) -> PyTree:
    """Preserve a pytree's values while stopping every leaf's gradient."""
    return jax.lax.stop_gradient(tree)
