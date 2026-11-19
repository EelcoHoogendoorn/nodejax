"""State-freezing transforms (`freeze` and `tree_freeze`) for inference and evaluation.

`freeze` converts a cyclic `Node` into an acyclic `Node` by evaluating every
`apply` call against a fixed `state` tree and discarding the state update.
`tree_freeze` applies `freeze` selectively across member subtrees addressed
either by a sparse `Struct` of frozen states or by a semantic `tag` (such as
`'running_stats'` on `BatchNorm`), leaving untagged dynamic carries (such as
recurrent hidden states) live.
"""

from __future__ import annotations

from nodejax.core.compose import _probe_apply
from nodejax.core.node import BaseNode, Node
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform


def _state_residue(node, state):
    if not node.cyclic:
        return ()
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        member = getattr(node.members, transparent)
        return _state_residue(member, state) if member.cyclic else state
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
    """Fixes `inner`'s state at `state`, discards step state updates, and returns an acyclic `Node`."""
    return _Freeze(inner, state)


def tree_freeze(node, frozen: Struct | None = None,
                tag: str | None = None) -> BaseNode:
    """Freezes selected member states of `node` by sparse `Struct` spec or leaf `tag`.

    Without `tag`, `frozen` supplies a sparse `Struct` whose fields match member
    paths to freeze while omitted cyclic members remain live. With `tag`,
    `frozen` supplies the full state tree (or is read directly from `node.state`
    when `node` is a `PSNode`) and every cyclic member declaring `tag` is
    frozen; when `node` starts as a `PSNode`, the returned view retains any
    unfrozen state residue as a `PSNode` (or becomes an acyclic `PNode` when
    no cyclic members remain).
    """
    if node.state_bound:
        rebuilt = tree_freeze(node.node, node.state if frozen is None else frozen,
                              tag=tag)
        bound = node.pnode._with_definition(rebuilt._def)
        if rebuilt.cyclic:
            return bound.bind(state=_state_residue(rebuilt, node.state))
        return bound
    if frozen is None:
        raise TypeError('tree_freeze needs the state to pin: a sparse spec, a '
                        'full state with tag=, or a state-bound node that '
                        'carries its own')
    return _tree_freeze_def(node, frozen, tag=tag)


@transform(preserves='param')
def _tree_freeze_def(node, frozen: Struct,
                     tag: str | None = None) -> Node:
    if tag is not None:
        derived = _tagged_spec(node, frozen, tag)
        if derived is None:
            raise ValueError(
                f"tree_freeze: tag {tag!r} matched nothing in '{node.name}'")
        frozen = derived
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        member = getattr(node.members, transparent)
        if node.cyclic and not member.cyclic:
            return freeze(node, frozen)
        rebuilt = tree_freeze(member, frozen)
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
            new[k] = m
        elif m.members:
            new[k] = tree_freeze(m, frozen[k])
        elif m.cyclic:
            new[k] = freeze(m, frozen[k])
        else:
            new[k] = m
    return Node(node._def.bind_members(Struct(**{
        name: member._def for name, member in new.items()})))


def _tagged_spec(node, state, tag: str):
    transparent = node._def.layout.transparent_member
    if transparent is not None:
        member = getattr(node.members, transparent)
        if member.cyclic:
            return _tagged_spec(member, state, tag)
        return state if (node.cyclic and tag in node.tags) else None
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
