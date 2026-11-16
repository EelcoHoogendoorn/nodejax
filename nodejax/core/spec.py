"""Constructs and manipulates input shape/dtype specifications (`jax.ShapeDtypeStruct` and `AxisSpec`).

Abstract shape propagation uses `jax.ShapeDtypeStruct` leaves for concrete shapes and `AxisSpec` nodes for leading mapped or scanned axes whose extent may still be unknown at assembly time. `materialize` converts abstract spec leaves into zero arrays while passing concrete array leaves through unchanged so input-primed Nodes keep real initial values when supplied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.types import PyTree
from nodejax.core.binding import AxisSpec, _spec_resolved


def spec(shape: int | tuple = (), dtype=jnp.float32) -> jax.ShapeDtypeStruct:
    """Creates a `jax.ShapeDtypeStruct`, treating an integer `shape` as a 1-D `(shape,)` tuple."""
    if type(shape) is int:
        shape = (shape,)
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def spec_of(tree: PyTree) -> PyTree:
    """Converts concrete leaves in `tree` to `jax.ShapeDtypeStruct` while preserving existing spec leaves."""
    return jax.tree.map(
        lambda leaf: leaf if type(leaf) in (jax.ShapeDtypeStruct, AxisSpec)
        else jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.result_type(leaf)),
        tree, is_leaf=lambda x: type(x) is AxisSpec)


def _axis_probe_spec(tree: PyTree) -> PyTree:
    """Expands `AxisSpec` nodes into temporary `ShapeDtypeStruct` leaves for `jax.eval_shape`."""
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
    """Replaces abstract spec leaves in `tree` with zero arrays while leaving concrete values intact."""
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
                fixed: bool = True) -> PyTree:
    """Wraps `spec` (field-wise across `Struct`s) in a leading `AxisSpec` of extent `n`."""
    if issubclass(type(spec), Struct):
        return Struct(**{
            k: add_axis(v, n, fixed=fixed) for k, v in spec.__items__})
    return AxisSpec(
        spec_of(spec) if _spec_resolved(spec) else spec, n, fixed=fixed) \
        if spec is not None else None


def axis_count(spec: PyTree) -> int | None:
    """Returns the leading `AxisSpec` count or leading concrete dimension of `spec`, or `None` if unknown."""
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
    """Peels one declared `AxisSpec` or concrete leading dimension off `spec`."""
    leaves = jax.tree.leaves(spec, is_leaf=lambda x: type(x) is AxisSpec)
    if any(type(l) is AxisSpec for l in leaves):
        # Unwrap AxisSpec subtrees in place while leaving non-axis sibling fields untouched.
        return jax.tree.map(
            lambda l: l.element if type(l) is AxisSpec else l,
            spec, is_leaf=lambda x: type(x) is AxisSpec)
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), spec_of(spec))
