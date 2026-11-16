"""Functional updates over keyed `Struct` and JAX PyTree paths.

Keyed PyTree registration (`Struct`, `PNode`, `PSNode`) assigns every leaf a dotted string path matching its attribute chain (`'.actuator.motor.resistance'`). `replace_by_path` updates existing leaves by value or `old -> new` callable, whereas `set_by_path` replaces entire `Struct` subtrees wholesale regardless of their previous structure.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax

from nodejax.core.types import PyTree


def set_by_path(tree: PyTree, updates: Mapping[str, PyTree]) -> PyTree:
    """Replaces the `Struct` fields or subtrees at each dotted path in `updates` wholesale."""
    from nodejax.struct import Struct

    def set_one(node, segments: list, value: PyTree, path: str) -> PyTree:
        head = segments[0]
        if not isinstance(node, Struct) or head not in node.__keys__:
            raise KeyError(f"path '{path}': no field '{head}'")
        if len(segments) == 1:
            return Struct(**{**dict(node.__items__), head: value})
        return Struct(**{**dict(node.__items__),
                         head: set_one(node[head], segments[1:], value, path)})

    for path, value in updates.items():
        tree = set_one(tree, path.strip('.').split('.'), value, path)
    return tree


def replace_by_path(tree: PyTree, updates: dict[str, PyTree]) -> PyTree:
    """Replaces each keyed leaf in `tree` matched by `updates` with a new value or `fn(old)` result."""
    keyed, treedef = jax.tree_util.tree_flatten_with_path(tree)
    remaining = dict(updates)
    leaves = []
    for key_path, leaf in keyed:
        path = jax.tree_util.keystr(key_path)
        if path in remaining:
            new = remaining.pop(path)
            leaf = new(leaf) if callable(new) else new
        leaves.append(leaf)
    if remaining:
        raise KeyError(f'paths not found in tree: {sorted(remaining)}')
    return jax.tree_util.tree_unflatten(treedef, leaves)
