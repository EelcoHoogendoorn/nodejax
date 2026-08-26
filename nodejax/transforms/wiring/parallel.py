from __future__ import annotations

from nodejax.core.composite import Composite
from nodejax.core.generic import Generic, is_generic
from nodejax.struct import Struct


_SEP = ' | '


def _name(names) -> str:
    return '(' + _SEP.join(names) + ')'


def parallel(**members):
    """Apply named members to matching fields of one Struct input."""
    if not members:
        raise TypeError('parallel needs at least one member')
    if any(is_generic(member) for member in members.values()):
        return Generic(
            _name(members), lambda **resolved: parallel(**resolved),
            Struct(**members))

    names = tuple(members)

    def param_fn(contract, param_input, rng):
        values = {}
        for name, member in contract.members.__items__:
            if member.parametric:
                values[name] = contract.member_param(
                    name, param_input[name], rng,
                    input_spec=contract.input_spec_for(name))
        return Struct(**values)

    def initialized(contract, param, state_input, rng, input=None):
        states = {}
        for name, member in contract.members.__items__:
            if not member.cyclic:
                continue
            options = ({'input': input[name]} if input is not None else
                       {'input_spec': contract.input_spec_for(name)})
            states[name] = contract.member_init(
                name,
                getattr(param, name) if member.parametric else (),
                state_input[name], rng, **options)
        return Struct(**states)

    def init_fn(contract, param, state_input, rng):
        return initialized(contract, param, state_input, rng)

    def prime_fn(contract, param, state_input, input, rng):
        return initialized(contract, param, state_input, rng, input)

    def apply_fn(contract, param, state, input, rng):
        states, outputs = {}, {}
        for name, member in contract.members.__items__:
            next_state, outputs[name] = member.apply(
                getattr(param, name) if member.parametric else (),
                getattr(state, name) if member.cyclic else (),
                member.feed(input[name]),
                rng.child(member.apply_takes_rng))
            if member.cyclic:
                states[name] = next_state
        return (Struct(**states) if states else ()), Struct(**outputs)

    input_spec = Struct(**{
        name: member.contract.input_spec
        for name, member in members.items()})
    has_param = any(member.parametric for member in members.values())
    has_state = any(member.cyclic for member in members.values())
    return Composite(**members).roles(
        apply_fn,
        param=param_fn if has_param else None,
        init=init_fn if has_state else None,
        prime=prime_fn if has_state else None,
        name=_name(names),
        apply_fields=names,
        input_spec=input_spec,
    )
