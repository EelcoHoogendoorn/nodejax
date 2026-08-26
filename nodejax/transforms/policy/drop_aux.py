"""Stop the aux stream here."""

from __future__ import annotations

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def drop_aux(inner: Node) -> Node:
    """Discard auxiliary values emitted at this node's output boundary."""
    def apply_fn(contract, param, state, input, rng):
        new_state, out = contract.members.inner.apply(
            param, state, input, rng)
        return new_state, split_aux(out)[0]

    return Wrapper(inner=inner).roles(
        name=f'drop_aux({inner.name})',
        apply=apply_fn,
    )
