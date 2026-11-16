"""Single-field `Struct` routing (`at`) for serial pipelines."""

from __future__ import annotations

from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.transform import transform


@transform(preserves='param')
def at(inner: Node, field: str) -> Node:
    """Applies `inner` to `input[field]` of an open `Struct` input and passes sibling fields through unchanged.

    Retains `inner`'s parameter tree, state tree, and `name` transparently so
    member keys and parameter paths in an enclosing `>>` pipeline read as if
    `inner` were placed directly.
    """

    def param(contract, param_input, rng):
        current = contract.members.inner.for_input(
            contract.input_spec_for(field))
        return current.param(param_input, rng)

    def init(contract, param, state_input, rng):
        current = contract.members.inner.for_input(
            contract.input_spec_for(field))
        return current.init(param, state_input, rng)

    def prime(contract, param, state_input, input, rng):
        return contract.members.inner.prime(
            param, state_input, input[field], rng)

    def apply(contract, param, state, input, rng):
        current = contract.members.inner
        next_state, output = current.apply(
            param, state, current.feed(input[field]), rng)
        return next_state, input.replace(**{field: output})

    return Wrapper(inner=inner).roles(
        name=inner.name,
        param=param,
        init=init,
        prime=prime,
        apply=apply,
        apply_fields=(field,),
        open=True,
    )
