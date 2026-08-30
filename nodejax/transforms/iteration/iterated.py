from __future__ import annotations

import jax

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def iterated(step: Node, n: int) -> Node:
    """Apply one node ``n`` times to the same input, threading state.

    Function iteration on the state channel: the input is held constant
    across iterations, unlike ``repeat``, which chains each output into
    the next input. The last iteration's output is returned, auxiliary
    outputs stack over iterations, and stochastic applications receive
    separate keys.
    """
    count = n

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.step
        rngs, _ = rng.axis(current.apply_takes_rng, count)

        def one(carry, child_rng):
            advanced, out = current.apply(param, carry, input, child_rng)
            clean, aux = split_aux(out)
            return advanced, (clean, aux)

        final, (outputs, auxs) = jax.lax.scan(
            one, state, rngs, length=count)
        last = jax.tree.map(lambda value: value[-1], outputs)
        return final, (last, auxs) if auxs is not None else last

    return Wrapper(step=step).roles(
        name=f'iterated({step.name})',
        apply=apply_fn,
    )
