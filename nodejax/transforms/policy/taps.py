from __future__ import annotations

from nodejax.core.binding import Aux, split_aux
from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def taps(inner: Node) -> Node:
    """Observe every wire of a composite def: the output becomes
    (final carry, Struct(<each member's output, keyed by name>)) — the aux
    convention with every member opted in. Because taps are ordinary
    outputs, batch/ensemble/scan add their axes to them automatically.
    Shallow: this pipe's wires, not nested ones — tap an inner pipe
    before composing to see inside it."""
    if inner._def.layout.kind != 'serial':
        raise TypeError(f'taps requires a serial pipe def (its members chain on '
                        f'the carry), got {inner!r}')

    def apply_fn(contract, param, state, input, rng):
        pipe = contract.members.inner
        members = pipe.members
        carry = pipe.intake(input)
        states, aux = {}, {}
        for name, member in members.__items__:
            child_rng = rng.child(member.apply_takes_rng)
            next_state, out = member.apply(
                getattr(param, name) if member.parametric else (),
                getattr(state, name) if member.cyclic else (),
                member.feed(carry), child_rng)
            if member.cyclic:
                states[name] = next_state
            carry, member_aux = split_aux(out)
            aux[name] = (carry if member_aux is None
                         else (carry, member_aux))
        return (Struct(**states) if states else ()), (carry, Aux(**aux))

    return Wrapper(inner=inner).roles(
        name=f'taps({inner.name})',
        apply=apply_fn,
    )
