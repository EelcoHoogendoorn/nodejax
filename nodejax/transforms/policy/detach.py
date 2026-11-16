"""Parameter gradient-stopping transforms (`detach` and `tree_detach`).

`detach` applies `jax.lax.stop_gradient` to an entire `Node`'s parameter tree
during `apply` while still allowing gradients with respect to runtime inputs and
state to flow through. `tree_detach` applies `detach` selectively to matching
member subtrees inside a composite `Node`.
"""

from __future__ import annotations

from typing import Callable

from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform
from nodejax.tree import tree_stop_gradient


@transform(preserves='param,state')
def detach(inner: Node) -> Node:
    """Stops gradients through `inner`'s parameter tree while leaving input and state gradients intact."""

    def apply_fn(contract, param, state, input, rng):
        return contract.members.inner.apply(
            tree_stop_gradient(param), state, input, rng)

    return Wrapper(inner=inner).roles(
        name=f'detach({inner.name})',
        apply=apply_fn,
    )


@transform(preserves='param,state')
def tree_detach(node, name: str | Callable) -> Node:
    """Wraps member subtrees of `node` matched by substring or predicate `name` in `detach`.

    Unlike `detach`, which stops gradients across a leaf or composite whole,
    `tree_detach` descends through non-matching composite members and raises
    `ValueError` if `name` matches zero members.
    """
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
