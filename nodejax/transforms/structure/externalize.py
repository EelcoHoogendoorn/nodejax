"""Move a parameter subtree into the runtime input."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax

from nodejax.core.definition import Def
from nodejax.core.node import Node
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform
from nodejax.paths import set_by_path


def _member_at(definition: Def, path: str) -> Def:
    """The member definition at ``path``, transparent levels passed through."""
    transparent = definition.layout.transparent_member
    if transparent is not None:
        return _member_at(getattr(definition.members, transparent), path)
    head, _, rest = path.partition('.')
    below = getattr(definition.members, head)
    return _member_at(below, rest) if rest else below


def _stand_in(contract) -> Any:
    """Values for an empty slot while state is built: the node's own
    parameterization from a fixed key. State shapes do not depend on them
    and nothing keeps them, so any values serve."""
    probe = MaybeKeyStream(jax.random.PRNGKey(0))
    return contract.param(Struct(), probe.child(contract.param_takes_rng))


def _marked(definition: Def, path: str) -> Def:
    """Record ``path`` as externalized at every level down to the member's
    parent, so any densification along the way accepts the empty slot.

    Transparent levels pass the path through unchanged; a composite level
    consumes its leading segment. The externalize wrapper itself carries
    the full path from its own construction."""
    transparent = definition.layout.transparent_member
    if transparent is not None:
        below = {transparent: _marked(getattr(definition.members, transparent), path)}
    else:
        head, _, rest = path.partition('.')
        below = ({head: _marked(getattr(definition.members, head), rest)}
                 if rest else {})
    if below:
        members = dict(definition.members.__items__)
        members.update(below)
        definition = definition.bind_members(Struct(**members))
    layout = replace(
        definition.layout,
        externalized_param_paths=definition.layout.externalized_param_paths | {path},
    )
    return definition.copy(layout=layout)


@transform
def externalize(inner: Node, member: str | None = None,
                at_init: Any | None = None, *,
                field: str | None = None) -> Node:
    """Demote a subtree from parameter to input: the member disappears from
    the node's param tree and becomes a required call field beside the
    node's own, named by the path's last segment. With no member named, the
    whole parameter tree is the input and ``field`` names the field it
    arrives in; the node is then not parametric at all.

    The dual of parameterize: parameterize demotes input kwargs to params;
    externalize promotes a param slot to input kwargs. Use when ONE
    instance of a composite must vary across batch elements (which
    ensemble cannot do, because ensemble stacks ALL members) or when
    one member's params must vary where the rest are shared — per-task
    worlds under a meta-learned controller, one model evaluated
    against many plants — the varying subtree has to leave the tree,
    and the only other channel is input. externalize is that move;
    stock batch is then correct again.

    Scoping falls out as a bonus, tuner-agnostically: a gradient tuner
    given this node's params has no `member` to differentiate, so it
    cannot adapt the world it is being adapted to. (Derivative-free
    optimizers scope by enumeration instead — dicts of paths, as in
    the actuator tuning workflow; gradient transforms consume whole
    trees, so they scope by tree structure.)

    tie's sibling in the reparameterization family: where tie shares a
    subtree within the tree, externalize hands one to the caller as
    data. The member's params are bound at apply. Init and prime run
    against ``at_init`` when supplied and otherwise against the member's
    own parameterization from a fixed key: a composite init that builds
    state by running its members needs values in the slot, and state
    shapes do not depend on which. param-rewriting: nodes only."""
    if not inner.parametric:
        raise TypeError('externalize requires a parametric node')

    if member is None:
        if field is None:
            raise TypeError('externalize of the whole node names its input field')
        return _externalize_root(inner, field, at_init)

    # `member` addresses the subtree, at any depth: 'motor', or 'inner.motor'
    # for a plant inside a loop. The INPUT field it rides in on is named by
    # the last segment, since a field name cannot carry a path.
    field = member.rsplit('.', 1)[-1]
    fields = _call_fields(inner, field)
    inner = Node(_marked(inner._def, member))

    def param_fn(contract, param_input, rng):
        """Build the inner's params from its own input evidence, then empty the slot."""
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return set_by_path(current.param(param_input, rng), {member: ()})

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.inner
        full = set_by_path(param, {member: input[field]})
        return current.apply(full, state, _inner_bundle(input, fields), rng)

    def init_param(param, current):
        stand_in = (at_init if at_init is not None
                    else _stand_in(_member_at(current._def, member).contract))
        return set_by_path(param, {member: stand_in})

    def init_fn(contract, param, state_input, rng):
        """Bind the inner from its own input evidence only."""
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.init(init_param(param, current), state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        """Prime the inner state from its real input value."""
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.prime(
            init_param(param, current), state_input,
            current.intake(_inner_bundle(input, fields)), rng)

    # The inner's definition accepts the empty slot, so a bound view may
    # descend into it with the wrapper's own parameter tree.
    return Wrapper(inner=inner).roles(
        name=f'externalize({inner.name})',
        externalized_param_paths=frozenset((member,)),
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        apply_fields=fields + (field,),
        input_spec=None,
    )


def _call_fields(inner: Node, field: str) -> tuple[str, ...]:
    """The inner's own call fields, which the parameter field joins flat."""
    fields = inner.contract.apply_fields
    if field in fields:
        raise TypeError(
            f'externalize({inner.name}): the parameter field {field!r} collides '
            f'with an input field of the same name')
    return fields


def _inner_bundle(input: Struct, fields: tuple[str, ...]) -> Struct:
    """The inner's formed call bundle from one externalized call."""
    return Struct(**{name: input[name] for name in fields})


def _inner_wire(contract, fields: tuple[str, ...]):
    """The inner's input evidence from the externalized node's, or None."""
    spec = contract.input_spec
    if spec is None:
        return None
    return contract.members.inner.intake(_inner_bundle(spec, fields))


def _externalize_root(inner: Node, field: str, at_init: Any | None) -> Node:
    """The whole parameter tree arrives in ``field``; the node has no params.

    Init and prime hand the inner ``at_init`` when given and its own
    parameterization from a fixed key otherwise."""
    fields = _call_fields(inner, field)

    def stand_in(current):
        return at_init if at_init is not None else _stand_in(current)

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.inner
        return current.apply(input[field], state, _inner_bundle(input, fields), rng)

    def init_fn(contract, param, state_input, rng):
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.init(stand_in(current), state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.prime(
            stand_in(current), state_input,
            current.intake(_inner_bundle(input, fields)), rng)

    return Wrapper(inner=inner).roles(
        name=f'externalize({inner.name})',
        destructurable=False,
        param=False,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        apply_fields=fields + (field,),
        input_spec=None,
    )
