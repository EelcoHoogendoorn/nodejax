"""Generic walks over the node and state trees.

map_members rewrites a node tree bottom-up, rebuilding each composite
from its recipe so flags recompute at every level — the reusable
structural-rewrite primitive. tree_filter prunes a state tree to the subtrees
under matching member names, producing the sparse Struct that
tree_freeze (and kin) consume.
"""

from __future__ import annotations

from typing import Callable

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, Composite
from nodejax.transforms.common import _over_bound


@_over_bound
def map_members(nd: NodeDef, fn: Callable[[NodeDef], NodeDef]) -> NodeDef:
    """Rewrite a node tree bottom-up: apply fn to every node, members
    first, rebuilding each composite from its recipe so cyclic/parametric
    recompute at each level. fn: NodeDef -> NodeDef."""
    if isinstance(nd, Composite):
        if nd.rebuild is None:
            raise TypeError(f"cannot rewrite '{nd.name}': no rebuild recipe")
        nd = nd.rebuild({k: map_members(m, fn) for k, m in nd.members.items()})
    return fn(nd)


def tree_filter(state: Struct, name: str | Callable[[str], bool]) -> Struct:
    """Prune a state tree to the members `name` matches: a matching key
    keeps its whole subtree, a subtree with matches nested below is kept
    pruned, the rest are dropped. The result is the sparse Struct spec
    tree_freeze consumes. `name` is a substring or a predicate on a key;
    match member names, not leaf fields. Matching nothing is an error: a
    filter that selects nothing was aimed at the wrong tree."""
    match = (lambda n: name in n) if isinstance(name, str) else name

    def prune(s: Struct) -> dict:
        out = {}
        for k, v in s.__items__:
            if match(k):
                out[k] = v
            elif isinstance(v, Struct):
                sub = prune(v)
                if sub:
                    out[k] = Struct(**sub)
        return out

    pruned = prune(state)
    if not pruned:
        raise ValueError(f'tree_filter: {name!r} matched nothing in the state tree')
    return Struct(**pruned)


def map_node_leaves(nd: NodeDef, fn: Callable[[NodeDef], Any]) -> Any:
    """Map `fn(leaf_node) -> value` over the leaf nodes of `nd`, building
    a matching Struct tree over composite member structures."""
    if not nd.cyclic:
        return 0
    if isinstance(nd, Composite):
        return Struct(**{nm: map_node_leaves(m, fn) for nm, m in nd.members.items()})
    return fn(nd)


def map_state_leaves(nd: NodeDef, state: Any, fn: Callable[[NodeDef, Any], Any]) -> Any:
    """Map `fn(leaf_node, leaf_state) -> leaf_state` over matching node and
    state tree hierarchies."""
    if not nd.cyclic or (isinstance(state, tuple) and len(state) == 0):
        return state
    if isinstance(nd, Composite):
        return Struct(**{nm: map_state_leaves(m, state[nm], fn)
                         for nm, m in nd.members.items() if nm in state})
    return fn(nd, state)
