"""Path-addressed pytree surgery.

Keyed registration (PNode, Struct, DQ, ...) gives every leaf a stable
string address ('.actuator.motor.resistance' — identical to the
attribute chain).
replace_by_path is the functional-update primitive over those
addresses (parameter genomes, domain randomization, targeted surgery).

Updates are a dict path -> new value, or path -> callable(old) -> new
(multiplicative domain factors, samplers). Unknown paths fail loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nodejax.core.types import PyTree

import jax


def set_by_path(tree: Any, updates: Mapping[str, Any]) -> Any:
    """Return `tree` with the SUBTREES at the given keyed paths replaced
    wholesale — the structural complement of replace_by_path's per-leaf
    edits. A path may address any Struct field at any depth, and the old
    subtree is discarded whatever its shape: a Struct may become (), an
    empty slot may become a Struct. Unknown paths fail loudly."""
    from nodejax.struct import Struct

    def set_one(node, segments: list, value: Any, path: str) -> Any:
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


def replace_by_path(tree: PyTree, updates: dict[str, Any]) -> PyTree:
    """Return `tree` with the leaves at the given keyed paths replaced.

    A value may be a callable, applied to the old leaf (enabling
    relative edits: lambda v: v * factor). Every path must match a leaf.
    """
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
