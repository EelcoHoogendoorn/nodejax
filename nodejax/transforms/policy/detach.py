"""Stop gradients through selected parameter trees."""

from __future__ import annotations

from typing import Callable

from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.tree import tree_stop_gradient
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param,state')
def detach(inner: Node) -> Node:
    """Stop gradient through a node's params: training leaves its weights
    fixed (they receive zero gradient), state and behaviour otherwise
    unchanged."""

    def apply_fn(contract, param, state, input, rng):
        return contract.members.inner.apply(
            tree_stop_gradient(param), state, input, rng)

    return Wrapper(inner=inner).roles(
        name=f'detach({inner.name})',
        apply=apply_fn,
    )


@transform(preserves='param,state')
def tree_detach(node, name: str | Callable) -> Node:
    """Detach the weights of members selected by key, descending into
    everything else. `name` is a substring or a predicate on a member key.
    Matching nothing is an error because the selection targeted the wrong
    tree."""
    match = (lambda n: name in n) if type(name) is str else name
    if not node.members:
        raise TypeError(f"tree_detach selects members and '{node.name}' has none; "
                        'detach(node) stops gradient through a leaf whole')
    rebuilt, hits = _detach_walk(node, match)
    if hits == 0:
        raise ValueError(f"tree_detach: {name!r} matched no member of '{node.name}'")
    return rebuilt


def _detach_walk(node, match: Callable) -> tuple[Node, int]:
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        res, hits = _detach_walk(getattr(node.members, transparent), match)
        return Node(node._def.bind_members(
            Struct(**{transparent: res._def}))), hits
    new, hits = {}, 0
    for k, m in node.members.__items__:
        if match(k):
            new[k] = detach(m)
            hits += 1
        elif m.members:
            new[k], sub = _detach_walk(m, match)
            hits += sub
        else:
            new[k] = m
    return Node(node._def.bind_members(Struct(**{
        name: member._def for name, member in new.items()}))), hits
