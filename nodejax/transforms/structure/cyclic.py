"""State-carry conversion (`cyclic`) turning a stateless transition step into a cyclic `Node`."""

from __future__ import annotations

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform


@transform(preserves='param,state')
def cyclic(step: Node) -> Node:
    """Moves `step`'s first declared `apply` field into cyclic `state` and threads its output as the next carry.

    Any remaining fields on `step` stay runtime `apply` inputs, while the first
    field's name becomes a required state-initialization field on `init(...)`
    (unless `step` already carries bound state). A single-argument `step(s) ->
    next_s` becomes a zero-input autonomous system that can step under `scan(
    ..., n=length)` or `repeat(..., n)`.
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
