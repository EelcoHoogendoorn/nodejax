from __future__ import annotations

import jax

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def repeat(layer: Node, n: int) -> Node:
    """Apply one node ``n`` times, sharing parameters and threading state.

    Outputs feed the next iteration. Auxiliary outputs stack over iterations,
    and stochastic applications receive separate keys.
    """
    count = n

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.layer
        rngs, _ = rng.axis(
            current.apply_takes_rng, count)
        first = current.intake(input)

        def step(carry, child_rng):
            s, x = carry
            s2, out = current.apply(
                param, s, current.feed(x), child_rng)
            clean, aux = split_aux(out)
            return (s2, clean), aux

        (s2, out), auxs = jax.lax.scan(
            step, (state, first), rngs, length=count)
        return s2, (out, auxs) if auxs is not None else out

    return Wrapper(layer=layer).roles(
        name=f'repeat({layer.name})',
        apply=apply_fn,
    )
