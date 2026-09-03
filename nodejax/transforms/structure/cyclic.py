"""A step promoted to a system: its first field becomes its state."""

from __future__ import annotations

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform


@transform(preserves='param,state')
def cyclic(step: Node) -> Node:
    """The cyclic form of a step whose output is the successor of its first
    field.

    The step's first field becomes the state, a call's output is that
    state's successor, and the remaining fields are the call's own. The
    initial condition is a state input under the first field's name, or a
    bound state. A step with one field gives a system with no call fields,
    ticking on its own. Parameters stay the step's.
    """
    if step.cyclic:
        raise TypeError(
            f"cyclic({step.name}) requires an acyclic step; "
            f"'{step.name}' is already cyclic"
        )
    fields = tuple(step.contract.apply_fields)
    if not fields:
        raise TypeError(f'cyclic({step.name}) needs a step with a first field to carry')
    carried, rest = fields[0], fields[1:]

    def init_fn(state_input):
        return state_input[carried]

    def apply_fn(contract, param, state, input, rng):
        bundle = Struct(**{carried: state, **{field: input[field] for field in rest}})
        _, output = contract.members.step.apply(param, (), bundle, rng)
        successor, aux = split_aux(output)
        return successor, (successor, aux) if aux is not None else successor

    return Wrapper(step=step).roles(
        name=f'cyclic({step.name})',
        init=init_fn,
        state_fields=(carried,),
        apply=apply_fn,
        apply_fields=rest,
        apply_takes_rng=step.contract.apply_takes_rng,
    )
