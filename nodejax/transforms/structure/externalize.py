"""Parameter-to-input routing (`externalize`) for per-call or per-batch varying weights.

`externalize` removes a member subtree (or the entire parameter tree) from a
`Node`'s stored `param` layout and appends a corresponding runtime `apply`
keyword field. Moving varying parameters into runtime inputs lets `batch` map a
single composite across per-element worlds or plants while keeping shared
controller parameters unmapped and excluding externalized parameters from
gradient optimizers acting on `.param`.
"""

from __future__ import annotations

from dataclasses import replace

import jax

from nodejax.core.definition import Def
from nodejax.core.node import Node
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.types import PyTree
from nodejax.core.wrapper import Wrapper
from nodejax.paths import set_by_path
from nodejax.struct import Struct
from nodejax.transforms.transform import transform


def _member_at(definition: Def, path: str) -> Def:
    transparent = definition.layout.transparent_member
    if transparent is not None:
        return _member_at(getattr(definition.members, transparent), path)
    head, _, rest = path.partition('.')
    below = getattr(definition.members, head)
    return _member_at(below, rest) if rest else below


def _stand_in(contract) -> PyTree:
    probe = MaybeKeyStream(jax.random.PRNGKey(0))
    return contract.param(Struct(), probe.child(contract.param_takes_rng))


def _marked(definition: Def, path: str) -> Def:
    """Marks `path` as externalized down the definition chain so densification accepts `()`."""
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
                at_init: PyTree = None, *,
                field: str | None = None) -> Node:
    """Moves `member`'s parameter subtree (or `inner`'s whole parameter tree) into a runtime `apply` field.

    When `member` names a dotted member path (such as `'motor'` or
    `'loop.motor'`), that subtree is replaced by `()` in `.param` and supplied
    at `apply` time under a keyword named after `member`'s final path segment.
    When `member` is `None`, `field` is required, the resulting `Node` becomes
    non-parametric, and the entire parameter tree arrives via `field` at `apply`
    time. State initialization (`init` and `prime`) substitutes `at_init` when
    provided or synthesizes temporary stand-in weights from a fixed key so
    member shape walks succeed.
    """
    if not inner.parametric:
        raise TypeError('externalize requires a parametric node')

    if member is None:
        if field is None:
            raise TypeError('externalize of the whole node names its input field')
        return _externalize_root(inner, field, at_init)

    # `member` may be a nested dotted path; the added `apply` field takes its final segment.
    field = member.rsplit('.', 1)[-1]
    fields = _call_fields(inner, field)
    inner = Node(_marked(inner._def, member))

    def param_fn(contract, param_input, rng):
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
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.init(init_param(param, current), state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        current = contract.members.inner.for_input(_inner_wire(contract, fields))
        return current.prime(
            init_param(param, current), state_input,
            current.intake(_inner_bundle(input, fields)), rng)

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
    fields = inner.contract.apply_fields
    if field in fields:
        raise TypeError(
            f'externalize({inner.name}): the parameter field {field!r} collides '
            f'with an input field of the same name')
    return fields


def _inner_bundle(input: Struct, fields: tuple[str, ...]) -> Struct:
    return Struct(**{name: input[name] for name in fields})


def _inner_wire(contract, fields: tuple[str, ...]):
    spec = contract.input_spec
    if spec is None:
        return None
    return contract.members.inner.intake(_inner_bundle(spec, fields))


def _externalize_root(inner: Node, field: str, at_init: PyTree) -> Node:
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
