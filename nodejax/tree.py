"""Axis slicing, reshaping, broadcasting, and gradient helpers over JAX PyTrees."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.types import PyTree


def tile(tree: PyTree, n: int) -> PyTree:
    """Broadcasts every leaf of `tree` along a new leading axis of length `n`."""
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n,) + jnp.shape(leaf)),
        tree,
    )


def _along(axis: int, index) -> tuple:
    return (slice(None),) * axis + (index,)


def tree_first(tree: PyTree, axis: int = 0) -> PyTree:
    """Selects index `0` along `axis` on every leaf of `tree`."""
    return jax.tree.map(lambda leaf: leaf[_along(axis, 0)], tree)


def tree_last(tree: PyTree, axis: int = 0) -> PyTree:
    """Selects index `-1` along `axis` on every leaf of `tree`."""
    return jax.tree.map(lambda leaf: leaf[_along(axis, -1)], tree)


def tree_tail(tree: PyTree, axis: int = 0) -> PyTree:
    """Slices `1:` along `axis` on every leaf of `tree`."""
    return jax.tree.map(lambda leaf: leaf[_along(axis, slice(1, None))], tree)


def tree_len(tree: PyTree) -> int:
    """Returns the shared leading-axis length across all leaves of `tree`."""
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
    """Reshapes the first `axes` leading dimensions of every leaf to `shape`."""
    return jax.tree.map(
        lambda leaf: leaf.reshape(shape + jnp.shape(leaf)[axes:]),
        tree,
    )


def tree_swap_axes(tree: PyTree, axis_a: int, axis_b: int) -> PyTree:
    """Swaps `axis_a` and `axis_b` on every leaf of `tree`."""
    return jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, axis_a, axis_b),
        tree,
    )


def tree_take(tree: PyTree, indices) -> PyTree:
    """Gathers entries along the leading axis of every leaf by `indices`."""
    return jax.tree.map(lambda leaf: leaf[indices], tree)


def tree_broadcast_axis(tree: PyTree, count: int, axis: int) -> PyTree:
    """Inserts and broadcasts a new axis of length `count` at `axis` on every leaf."""
    def broadcast(leaf):
        shape = jnp.shape(leaf)
        normalized_axis = axis if axis >= 0 else len(shape) + axis + 1
        expanded = jnp.expand_dims(leaf, normalized_axis)
        target = shape[:normalized_axis] + (count,) + shape[normalized_axis:]
        return jnp.broadcast_to(expanded, target)

    return jax.tree.map(broadcast, tree)


def tree_stop_gradient(tree: PyTree) -> PyTree:
    """Stops gradients across every leaf of `tree` while preserving values."""
    return jax.lax.stop_gradient(tree)
