"""Move a parameter subtree into the runtime input."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from nodejax.core.definition import Def
from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform
from nodejax.paths import set_by_path


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
    the node's param tree and becomes a required field on `input`. With no
    member named, the whole parameter tree is the input and ``field`` names
    the field it arrives in; the node is then not parametric at all.

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
    data. The member's params are bound at apply; init runs against
    `at_init` when supplied — a composite init that spec-propagates by
    running its members (a persisted scan makes the outer init real)
    needs values in the slot, and since state shapes are independent
    of them, the member param operation's input defaults are the natural
    stand-in. With at_init omitted the slot stays empty at init,
    sufficient for inits that read shapes alone. param-rewriting:
    nodes only."""
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
    inner = Node(_marked(inner._def, member))

    def param_fn(contract, param_input, rng):
        """Build the inner's params from `.input` shape evidence, then empty the slot."""
        current = contract.members.inner.for_input(
            contract.input_spec_for('input'))
        return set_by_path(current.param(param_input, rng), {member: ()})

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.inner
        full = set_by_path(param, {member: input[field]})
        return current.apply(
            full, state, current.feed(input.input), rng)

    def init_param(param):
        return (set_by_path(param, {member: at_init})
                if at_init is not None else param)

    def init_fn(contract, param, state_input, rng):
        """Bind the inner from `.input` shape evidence only."""
        current = contract.members.inner.for_input(
            contract.input_spec_for('input'))
        return current.init(init_param(param), state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        """Prime the inner state from the real `.input` value."""
        return contract.members.inner.prime(
            init_param(param), state_input, input.input, rng)

    # The inner's definition accepts the empty slot, so a bound view may
    # descend into it with the wrapper's own parameter tree.
    return Wrapper(inner=inner).roles(
        name=f'externalize({inner.name})',
        externalized_param_paths=frozenset((member,)),
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        apply_fields=('input', field),
        input_spec=None,
    )


def _externalize_root(inner: Node, field: str, at_init: Any | None) -> Node:
    """The whole parameter tree arrives in ``field``; the node has no params.

    Init and prime hand the inner ``at_init`` when given and the empty tree
    otherwise, which serves any inner whose init reads shapes alone."""
    stand_in = () if at_init is None else at_init

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.inner
        return current.apply(
            input[field], state, current.feed(input.input), rng)

    def init_fn(contract, param, state_input, rng):
        current = contract.members.inner.for_input(
            contract.input_spec_for('input'))
        return current.init(stand_in, state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        return contract.members.inner.prime(
            stand_in, state_input, input.input, rng)

    return Wrapper(inner=inner).roles(
        name=f'externalize({inner.name})',
        destructurable=False,
        param=False,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        apply_fields=('input', field),
        input_spec=None,
    )
