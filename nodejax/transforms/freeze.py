"""Freeze state or stop gradients through selected parameters."""

from __future__ import annotations

from typing import Callable

import jax

from nodejax.compose import _probe_apply
from nodejax.node import Node
from nodejax.struct import Struct
from nodejax.transform import transform
from nodejax.wrapper import Wrapper


def _state_residue(node, state):
    if not node.cyclic:
        return ()
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        return _state_residue(getattr(node.members, transparent), state)
    if node.members:
        return Struct(**{
            name: _state_residue(child, state[name])
            for name, child in node.members.__items__
            if child.cyclic and name in state
        })
    return state


def _Freeze(inner: Node, state) -> Node:
    def apply_fn(contract, param, input, rng):
        current = contract.members.inner
        _, out = _probe_apply(
            current.apply, param, state, input, rng)
        return out

    return Wrapper(inner=inner).roles(
        name=f'freeze({inner.name})',
        apply=apply_fn, init=False,
    )


@transform(preserves='param')
def freeze(inner: Node, state) -> Node:
    """Hold a node's whole state fixed: the result applies with `state`,
    discards the returned update, and is non-cyclic. Params unchanged."""
    return _Freeze(inner, state)


@transform(preserves='param,state')
def detach(inner: Node) -> Node:
    """Stop gradient through a node's params: training leaves its weights
    fixed (they receive zero gradient), state and behaviour otherwise
    unchanged."""

    def apply_fn(contract, param, state, input, rng):
        return contract.members.inner.apply(
            jax.lax.stop_gradient(param), state, input, rng)

    return Wrapper(inner=inner).roles(
        name=f'detach({inner.name})',
        apply=apply_fn,
    )


@transform(preserves='param,state')
def tree_detach(node, name: str | Callable) -> Node:
    """Detach the weights of the members `name` matches (by key),
    descending into the rest. The param-side twin of tree_freeze — but
    it selects by name directly rather than by a spec, because detaching
    needs no state values, only which nodes. `name`: substring or a
    predicate on a member key. Matching nothing is an error: a selective
    walk that selects nothing was aimed at the wrong tree."""
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
            new[k] = detach(m)                      # detach the matching subtree
            hits += 1
        elif m.members:
            new[k], sub = _detach_walk(m, match)    # descend
            hits += sub
        else:
            new[k] = m
    return Node(node._def.bind_members(Struct(**{
        name: member._def for name, member in new.items()}))), hits


def tree_freeze(node, frozen: Struct | None = None,
                tag: str | None = None):
    """Freeze the members present in `frozen` (a sparse Struct spec): a
    composite member descends, a leaf member is frozen with its state, an
    absent member stays threaded. Composites rebuild, so cyclicity
    propagates. Every spec field must land on a member — a field naming
    nothing is an error, so a spec aimed at structure a node does not
    expose fails at the call, never later inside an apply.

    With `tag`, `frozen` is the complete state and the sparse spec derives
    from the node tree: every member whose leaf def declares the tag
    freezes at its state. Selection by what the state IS (running_stats),
    not by what a layer happens to be called.

    A state-bound node consumes its own binding: tree_freeze(model, tag=...)
    reads the state it holds, rebuilds the node, and returns a state-bound
    node again, carrying exactly the state that still moves. A whole-tree
    freeze leaves it stateless: the successor of every apply then equals
    its predecessor, and discarding it is finally safe."""
    if node.state_bound:
        rebuilt = tree_freeze(node.node, node.state if frozen is None else frozen,
                              tag=tag)
        return node.pnode._with_definition(rebuilt._def).bind(
            state=_state_residue(rebuilt, node.state))
    if frozen is None:
        raise TypeError('tree_freeze needs the state to pin: a sparse spec, a '
                        'full state with tag=, or a state-bound node that '
                        'carries its own')
    return _tree_freeze_def(node, frozen, tag=tag)


@transform(preserves='param')
def _tree_freeze_def(node, frozen: Struct,
                     tag: str | None = None) -> Node:
    """tree_freeze's def-level worker; the public entry dispatched the
    state-bound case and guaranteed `frozen`."""
    if tag is not None:
        derived = _tagged_spec(node, frozen, tag)
        if derived is None:
            raise ValueError(
                f"tree_freeze: tag {tag!r} matched nothing in '{node.name}'")
        frozen = derived
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        rebuilt = tree_freeze(getattr(node.members, transparent), frozen)
        return Node(node._def.bind_members(
            Struct(**{transparent: rebuilt._def})))
    if not node.members:
        raise TypeError(f"tree_freeze selects members and '{node.name}' has none; "
                        'freeze(node, state) pins a leaf whole')
    present = {k for k, _ in frozen.__items__}
    unknown = present - set(node.members.__keys__)
    if unknown:
        raise TypeError(f"tree_freeze: {sorted(unknown)} name no member of '{node.name}'")
    new = {}
    for k, m in node.members.__items__:
        if k not in present:
            new[k] = m                              # absent -> keep threaded
        elif m.members:
            new[k] = tree_freeze(m, frozen[k])            # composite -> descend
        elif m.cyclic:
            new[k] = freeze(m, frozen[k])                 # leaf -> freeze whole
        else:
            new[k] = m
    return Node(node._def.bind_members(Struct(**{
        name: member._def for name, member in new.items()})))


def _tagged_spec(node, state, tag: str):
    """The sparse spec tree_freeze consumes, derived from a tag: a field
    for every member subtree whose leaf def declares it, filled with that
    member's state. None where nothing beneath declares."""
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        return _tagged_spec(getattr(node.members, transparent), state, tag)
    if not node.members:
        return state if (node.cyclic and tag in node.tags) else None
    out = {}
    for k, m in node.members.__items__:
        if k not in state:
            continue
        sub = _tagged_spec(m, state[k], tag)
        if sub is not None:
            out[k] = sub
    return Struct(**out) if out else None
