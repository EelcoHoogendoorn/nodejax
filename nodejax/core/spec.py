"""Input-shape declarations and derived shape operations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.types import PyTree
from nodejax.core.binding import AxisSpec, _spec_resolved


def spec(shape: int | tuple = (), dtype: Any = jnp.float32) -> jax.ShapeDtypeStruct:
    """Declare a single array spec; an int is shorthand for a 1-d shape."""
    if type(shape) is int:
        shape = (shape,)
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def spec_of(tree: PyTree) -> PyTree:
    """The spec of a pytree: ShapeDtypeStruct leaves for concrete values;
    existing spec leaves, AxisSpec included, pass through."""
    return jax.tree.map(
        lambda leaf: leaf if type(leaf) in (jax.ShapeDtypeStruct, AxisSpec)
        else jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.result_type(leaf)),
        tree, is_leaf=lambda x: type(x) is AxisSpec)


def _axis_probe_spec(tree: PyTree) -> PyTree:
    """A concrete abstract representative of a spec that may declare an axis.

    JAX shape evaluation needs array-shaped leaves, while ``AxisSpec`` is a
    partial declaration rather than an array.  Replace each declared axis with
    a leading abstract axis: its declared extent when fixed and known, or one
    representative element otherwise.  Nested axes recurse, and an axis over a
    Struct distributes over its fields just like ``add_axis`` does.

    This is a throwaway shape probe.  It never changes the declaration or
    teaches a variable map the representative extent.
    """
    def expand(leaf):
        if type(leaf) is not AxisSpec:
            return leaf
        element = _axis_probe_spec(leaf.element)
        count = leaf.count if leaf.fixed and leaf.count is not None else 1
        return jax.tree.map(
            lambda item: jax.ShapeDtypeStruct(
                (count, *item.shape), item.dtype),
            element,
        )

    return jax.tree.map(
        expand, spec_of(tree), is_leaf=lambda x: type(x) is AxisSpec)


def materialize(tree: PyTree) -> PyTree:
    """Turn a spec (or value) into a materialized value: spec leaves become zeros,
    concrete leaves pass through unchanged — so a real value primes
    state with data where a spec only supplies shape. A declared axis
    materializes at its count, and refuses while the count is unknown."""
    def leaf(l):
        if type(l) is AxisSpec:
            if l.count is None:
                raise TypeError(
                    f'an axis with an unknown count cannot materialize: '
                    f'{l!r}; bind an input, or construct with the count')
            return jax.tree.map(
                lambda e: jnp.zeros((l.count, *e.shape), e.dtype),
                spec_of(materialize(l.element)))
        return (jnp.zeros(l.shape, l.dtype)
                if type(l) is jax.ShapeDtypeStruct else l)
    return jax.tree.map(leaf, tree, is_leaf=lambda x: type(x) is AxisSpec)


def add_axis(spec: PyTree, n: int | None = None, *,
                fixed: bool = True) -> AxisSpec:
    """Add a declared leading axis, optionally with a fixed extent."""
    if issubclass(type(spec), Struct):
        return Struct(**{
            k: add_axis(v, n, fixed=fixed) for k, v in spec.__items__})
    return AxisSpec(
        spec_of(spec) if _spec_resolved(spec) else spec, n, fixed=fixed) \
        if spec is not None else None


def axis_count(spec: PyTree) -> int | None:
    """The declared axis extent of a bundle spec: the first populated field's
    count (every field declaring one shares the axis), or the leading data
    axis of a resolved spec without one. None when nothing says."""
    def lead(tree):
        leaves = jax.tree.leaves(
            tree,
            is_leaf=lambda value: type(value) is AxisSpec,
        )
        if not leaves:
            return None
        leaf = leaves[0]
        if type(leaf) is AxisSpec:
            return leaf.count
        return leaf.shape[0] if leaf.shape else None

    if spec is None:
        return None
    return spec.count if type(spec) is AxisSpec else lead(spec)


def element_spec(spec: PyTree) -> PyTree:
    """Remove one declared or concrete leading axis from a spec."""
    leaves = jax.tree.leaves(spec, is_leaf=lambda x: type(x) is AxisSpec)
    if any(type(l) is AxisSpec for l in leaves):
        # axis subtrees unwrap in place; fields beside the axis (a boundary
        # key) pass unchanged, since the map never covered them
        return jax.tree.map(
            lambda l: l.element if type(l) is AxisSpec else l,
            spec, is_leaf=lambda x: type(x) is AxisSpec)
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), spec_of(spec))
