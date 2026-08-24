from __future__ import annotations

import jax

from nodejax.composite import Composite
from nodejax.generic import Generic, is_generic
from nodejax.struct import Struct


_SEP = ' + '


def _name(names) -> str:
    return '(' + _SEP.join(names) + ')'


def sum_junction(**members):
    """Broadcast one input to named members and add their outputs."""
    if not members:
        raise TypeError('sum_junction needs at least one member')
    if any(is_generic(member) for member in members.values()):
        return Generic(
            _name(members), lambda **resolved: sum_junction(**resolved),
            Struct(**members))

    names = tuple(members)

    def param_fn(contract, param_input, rng):
        values = {}
        for name, member in contract.members.__items__:
            if member.parametric:
                spec = contract.input_spec
                values[name] = contract.member_param(
                    name, param_input[name], rng,
                    input_spec=(None if spec is None else
                                contract.intake(spec)))
        return Struct(**values)

    def initialized(contract, param, state_input, rng, input=None):
        states = {}
        signal = input
        spec = contract.input_spec
        for name, member in contract.members.__items__:
            if not member.cyclic:
                continue
            options = ({'input': signal} if input is not None else
                       {'input_spec': (
                           None if spec is None else
                           contract.intake(spec))})
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
        signal = contract.intake(input)
        states, outputs = {}, []
        for name, member in contract.members.__items__:
            next_state, output = member.apply(
                getattr(param, name) if member.parametric else (),
                getattr(state, name) if member.cyclic else (),
                member.feed(signal),
                rng.child(member.apply_takes_rng))
            if member.cyclic:
                states[name] = next_state
            outputs.append(output)
        return (Struct(**states) if states else ()), jax.tree.map(
            lambda *terms: sum(terms), *outputs)

    head = members[names[0]]
    has_param = any(member.parametric for member in members.values())
    has_state = any(member.cyclic for member in members.values())
    return Composite(**members).roles(
        apply_fn,
        param=param_fn if has_param else None,
        init=init_fn if has_state else None,
        prime=prime_fn if has_state else None,
        name=_name(names),
        input_contract=head.contract,
    )
