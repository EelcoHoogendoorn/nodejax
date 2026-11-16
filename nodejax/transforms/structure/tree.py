"""Bottom-up definition rewriting (`map_members`) and state-subtree filtering (`tree_filter`)."""

from __future__ import annotations

from typing import Callable

from nodejax.core.node import Node, _is_node
from nodejax.struct import Struct


def map_members(node, fn: Callable) -> Node:
    """Rewrites an unbound `Node` tree bottom-up by applying `fn` to every rebuilt `Node`."""
    if not _is_node(node) or node.bound:
        raise TypeError(
            'map_members rewrites definitions; use bound.node explicitly '
            'and rebind any compatible data yourself')
    if node.members:
        members = {
            key: map_members(member, fn)
            for key, member in node.members.__items__
        }
        node = Node(node._def.bind_members(Struct(**{
            name: member._def for name, member in members.items()
        })))
    result = fn(node)
    if not _is_node(result) or result.bound:
        raise TypeError('map_members callback must return a Node')
    return result


def tree_filter(state, name: str | Callable) -> Struct:
    """Extracts the sparse `Struct` of `state` subtrees whose member keys match `name`."""
    match = (lambda member: name in member) if type(name) is str else name

    def prune(state: Struct) -> Struct:
        selected = {}
        for member, value in state.__items__:
            if match(member):
                selected[member] = value
            elif type(value) is Struct:
                nested = prune(value)
                if nested:
                    selected[member] = nested
        return Struct(**selected)

    selected = prune(state)
    if not selected:
        raise ValueError(
            f'tree_filter: {name!r} matched nothing in the state tree')
    return selected
