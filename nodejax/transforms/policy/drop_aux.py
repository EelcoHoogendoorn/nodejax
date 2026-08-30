"""Stop aux at a Node boundary or remove it from a runtime output."""

from __future__ import annotations

from nodejax.core.binding import split_aux
from nodejax.core.generic import Generic, is_generic
from nodejax.core.node import Node, _is_node
from nodejax.struct import Struct
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def _drop_aux_node(inner: Node) -> Node:
    """Discard auxiliary values emitted at this node's output boundary."""
    def apply_fn(contract, param, state, input, rng):
        new_state, out = contract.members.inner.apply(
            param, state, input, rng)
        value, aux = split_aux(out)
        return new_state, value

    return Wrapper(inner=inner).roles(
        name=f'drop_aux({inner.name})',
        apply=apply_fn,
    )


def drop_aux(value):
    """Discard Aux from a Node boundary or one concrete runtime output."""
    if is_generic(value):
        return Generic('drop_aux', _drop_aux_node, Struct(inner=value))
    if _is_node(value):
        return _drop_aux_node(value)
    output, aux = split_aux(value)
    return output
