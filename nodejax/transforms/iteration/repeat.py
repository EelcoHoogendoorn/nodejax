from __future__ import annotations

import jax

from nodejax.core.binding import split_aux
from nodejax.core.node import Node
from nodejax.transforms.iteration.scan import (
    _given_start, _run_takes_rng, _split_state_fields,
)
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param,state')
def repeat(layer: Node, n: int) -> Node:
    """Apply one node ``n`` times, sharing parameters and threading state.

    Outputs feed the next iteration. Auxiliary outputs stack over iterations,
    and stochastic applications receive separate keys. A state-bound layer
    runs from its bound state.
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
    """Apply one node ``n`` times from the input with fresh state on every
    call, each output feeding the next input, and return every output
    stacked over a leading axis.

    The past-complete form of ``repeat``, as ``scanned`` is of ``scan``: the
    trajectory of a step from a start, every point kept and the state
    discarded. A state-bound step runs from its bound state; otherwise the
    state is built from the step's state-input fields, which the call takes
    beside the start, and from the start itself. Parameters are shared,
    auxiliary outputs stack over the steps, and stochastic applications
    receive separate keys.
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
            initial = _given_start(current, state, rng)
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
