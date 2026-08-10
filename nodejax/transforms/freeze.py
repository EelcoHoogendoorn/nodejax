"""Hold things fixed: freeze/tree_freeze pin STATE, detach pins WEIGHTS.

freeze pins a whole node's state and drops its cyclic slot. tree_freeze
does it selectively from a SPARSE spec — a Struct with a field only for
each member to freeze: a field holding that node's state freezes it
whole, a field holding a sub-spec Struct descends, an absent field stays
threaded. So freezing one node by hand is `tree_freeze(model,
Struct(norm=state.norm))`, a full state freezes everything, and
tree_filter builds a partial spec by name. Composites are rebuilt from
their members, so cyclicity propagates (a composite whose stateful
members all froze recomputes to non-cyclic).

detach is the param-side twin: it stops gradient through a node's
weights, so training leaves them fixed. Selective weight-freezing is
detach composed with the map_members walk.
"""

from __future__ import annotations

from typing import Callable

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, Composite, _trivial_init_fn
from nodejax.transforms.common import _over_bound, _transform_def


def _Freeze(node_def: NodeDef, state) -> NodeDef:
    def apply(nd, p, _, input):
        _, out = node_def.apply_fn(p, state, input)
        return (), out

    return _transform_def(
        node_def,
        name=f'freeze({node_def.name})',
        init_fn=_trivial_init_fn,
        apply_fn=apply,
        cyclic=False,
        rebuild=lambda d: _Freeze(d, state),
    )


@_over_bound
def freeze(node_def: NodeDef, state) -> NodeDef:
    """Hold a node's whole state fixed: the result applies with `state`,
    discards the returned update, and is non-cyclic. Params unchanged."""
    return _Freeze(node_def, state)


@_over_bound
def detach(node_def: NodeDef) -> NodeDef:
    """Stop gradient through a node's params: training leaves its weights
    fixed (they receive zero gradient), state and behaviour otherwise
    unchanged."""

    def apply(nd, p, s, i):
        return node_def.apply_fn(jax.lax.stop_gradient(p), s, i)

    return _transform_def(
        node_def,
        name=f'detach({node_def.name})',
        apply_fn=apply,
        rebuild=lambda d: detach(d),
    )


@_over_bound
def tree_detach(nd: NodeDef, name: str | Callable[[str], bool]) -> NodeDef:
    """Detach the weights of the members `name` matches (by key),
    descending into the rest. The param-side twin of tree_freeze — but
    it selects by name directly rather than by a spec, because detaching
    needs no state values, only which nodes. `name`: substring or a
    predicate on a member key. Matching nothing is an error: a selective
    walk that selects nothing was aimed at the wrong tree."""
    match = (lambda n: name in n) if isinstance(name, str) else name
    if not isinstance(nd, Composite):
        raise TypeError(f"tree_detach selects members and '{nd.name}' has none; "
                        'detach(node) stops gradient through a leaf whole')
    rebuilt, hits = _detach_walk(nd, match)
    if hits == 0:
        raise ValueError(f"tree_detach: {name!r} matched no member of '{nd.name}'")
    return rebuilt


def _detach_walk(nd: Composite, match: Callable[[str], bool]) -> tuple[NodeDef, int]:
    new, hits = {}, 0
    for k, m in nd.members.items():
        if match(k):
            new[k] = detach(m)                      # detach the matching subtree
            hits += 1
        elif isinstance(m, Composite):
            new[k], sub = _detach_walk(m, match)    # descend
            hits += sub
        else:
            new[k] = m
    return nd.rebuild(new), hits


@_over_bound
def tree_freeze(nd: NodeDef, frozen: Struct) -> NodeDef:
    """Freeze the members present in `frozen` (a sparse Struct spec): a
    composite member descends, a leaf member is frozen with its state, an
    absent member stays threaded. Composites rebuild, so cyclicity
    propagates. Every spec field must land on a member — a field naming
    nothing is an error, so a spec aimed at structure a def does not
    expose fails at the call, never later inside an apply."""
    if not isinstance(nd, Composite):
        raise TypeError(f"tree_freeze selects members and '{nd.name}' has none; "
                        'freeze(node, state) pins a leaf whole')
    present = {k for k, _ in frozen.__items__}
    unknown = present - set(nd.members)
    if unknown:
        raise TypeError(f"tree_freeze: {sorted(unknown)} name no member of '{nd.name}'")
    new = {}
    for k, m in nd.members.items():
        if k not in present:
            new[k] = m                              # absent -> keep threaded
        elif isinstance(m, Composite):
            new[k] = tree_freeze(m, frozen[k])            # composite -> descend
        elif m.cyclic:
            new[k] = freeze(m, frozen[k])                 # leaf -> freeze whole
        else:
            new[k] = m
    return nd.rebuild(new)
