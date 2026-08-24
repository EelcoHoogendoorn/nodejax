from __future__ import annotations

from nodejax.node import Node
from nodejax.transform import transform
from nodejax.wrapper import Wrapper


@transform(preserves='param')
def at(inner: Node, field: str) -> Node:
    """Route a node onto one field of a Struct input: the output is the
    input Struct with `field` replaced by the node's output, every
    other field passed through untouched. Pipes chain whole signals;
    at() lets a chain act on one strand of a structured signal while
    the rest ride alongside.

    Type-preserving and transparent: param and state are the inner
    node's own, a supplied init input is projected to the field, and
    the wrapper keeps the inner node's name, so pipe member keys and
    state paths read as if the node were placed directly."""

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
