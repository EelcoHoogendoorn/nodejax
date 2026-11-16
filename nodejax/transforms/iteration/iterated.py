"""Fixed-count state iteration (`iterated`) with constant input."""

from __future__ import annotations

import jax

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.transform import transform


@transform(preserves='param')
def iterated(step: Node, n: int) -> Node:
    """Advances `step`'s state `n` times while holding the runtime `input` constant.

    Unlike `repeat`, which feeds each step's output into the next step's input,
    `iterated` supplies the same `input` on all `n` substeps, returns the final
    substep's output alongside stacked `Aux` values, and splits PRNG keys across
    substeps.
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
