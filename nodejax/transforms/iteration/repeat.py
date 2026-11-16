"""Weight-tied output-chaining iteration (`repeat` and `repeated`).

Both transforms share one parameter tree across `n` sequential applications
where each step's output becomes the next step's input. `repeat` threads state
externally (`preserves='param,state'`) and returns the final step's output,
whereas `repeated` internalizes state for one trajectory (`internalizes='state'`)
and returns all `n` step outputs stacked along leading axis 0.
"""

from __future__ import annotations

import jax

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.iteration.scan import (
    _run_takes_rng, _split_state_fields,
)
from nodejax.transforms.transform import transform


@transform(preserves='param,state')
def repeat(layer: Node, n: int) -> Node:
    """Chains `layer` `n` times with shared parameters, feeding each output into the next step.

    Preserves both parameter and state bindings (`preserves='param,state'`),
    returns the `n`-th output alongside any `Aux` values stacked along leading
    axis 0, and splits PRNG keys per step. Unlike `iterated`, which holds the
    input fixed while advancing state, `repeat` feeds the running output
    forward; unlike `repeated`, it keeps state external and returns only the
    final output.
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


@transform(preserves='param', internalizes='state')
def repeated(step: Node, n: int) -> Node:
    """Chains `step` `n` times from fresh or bound state and stacks all `n` outputs along axis 0.

    When `step` is a `PSNode`, its bound state seeds the trajectory; otherwise
    initial state is primed from the call's state-input fields and starting
    input, and the final state is discarded after collecting the `n`-step
    output trajectory.
    """
    if type(n) is not int or n < 1:
        raise TypeError(f'repeated n must be a positive int, got {n!r}')
    count = n
    state = step.state if step.state_bound else None
    step = step.node
    fields = () if state is not None else step.contract.state_input_fields

    def parameterize(contract, param_input, rng):
        current = contract.members.step
        spec = contract.input_spec
        if spec is not None:
            _, spec = _split_state_fields(spec, fields)
            current = current._resolve_def(spec, bundled=True).contract
        return current.param(param_input, rng)

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, start = _split_state_fields(input, fields)
        first = current.intake(start)
        if state is not None:
            initial = current._dense_state(state)
        elif current.cyclic:
            initial = current.prime(
                param, state_input, first, rng.child(current.init_takes_rng))
        else:
            initial = ()
        rngs, _ = rng.axis(current.apply_takes_rng, count)

        def advance(carry, child_rng):
            carried_state, x = carry
            next_state, out = current.apply(
                param, carried_state, current.feed(x), child_rng)
            clean, aux = split_aux(out)
            return (next_state, clean), (clean, aux)

        _, (outputs, auxs) = jax.lax.scan(
            advance, (initial, first), rngs, length=count)
        return (outputs, auxs) if auxs is not None else outputs

    form = ({'apply_fields': tuple(step.contract.apply_fields) + fields,
             'input_spec': None} if fields else {})
    return Wrapper(step=step).roles(
        name=f'repeated({step.name})',
        param=parameterize,
        init=False,
        apply=apply_fn,
        **form,
        apply_takes_rng=_run_takes_rng(step.contract, state),
    )
