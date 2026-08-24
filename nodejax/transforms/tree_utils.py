"""Generic rewrite and filter walks over node and state trees.

map_members rewrites a node tree bottom-up, rebuilding each composite
from its member nodes so flags recompute at every level: the reusable
structural-rewrite primitive. tree_filter prunes a state tree to the
subtrees under matching member names, producing the sparse Struct that
tree_freeze (and kin) consume. tile broadcasts an entire pytree along a
new leading axis for scanned batch replay.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import PyTree
from nodejax.node import Node, _is_node


def map_members(node, fn: Callable) -> Node:
    """Rewrite a node tree bottom-up: apply fn to every node, members
    first, rebuilding each composite from its member nodes so cyclic/parametric
    recompute at each level. fn: Node -> Node."""
    if not _is_node(node) or node.bound:
        raise TypeError(
            'map_members rewrites definitions; use bound.node explicitly '
            'and rebind any compatible data yourself')
    if node.members:
        members = {key: map_members(member, fn)
                   for key, member in node.members.__items__}
        node = Node(node._def.bind_members(Struct(**{
            name: member._def for name, member in members.items()})))
    result = fn(node)
    if not _is_node(result) or result.bound:
        raise TypeError('map_members callback must return a Node')
    return result


def tree_filter(state, name: str | Callable) -> Struct:
    """Prune a state tree to the members `name` matches: a matching key
    keeps its whole subtree, a subtree with matches nested below is kept
    pruned, the rest are dropped. The result is the sparse Struct spec
    tree_freeze consumes. `name` is a substring or a predicate on a key;
    match member names, not leaf fields. Matching nothing is an error: a
    filter that selects nothing was aimed at the wrong tree."""
    match = (lambda n: name in n) if type(name) is str else name

    def prune(s: Struct) -> Struct:
        out = {}
        for k, v in s.__items__:
            if match(k):
                out[k] = v
            elif type(v) is Struct:
                sub = prune(v)
                if sub:
                    out[k] = sub
        return Struct(**out)

    pruned = prune(state)
    if not pruned:
        raise ValueError(f'tree_filter: {name!r} matched nothing in the state tree')
    return pruned


def tile(tree: PyTree, n: int) -> PyTree:
    """Tile a pytree along a new leading axis: the spelling for scanning
    a fixed batch n times."""
    return jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (n,) + jnp.shape(leaf)), tree)
