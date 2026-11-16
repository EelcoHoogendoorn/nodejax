"""Sequence-axis transforms (`scan`, `scanned`, and `carried`) over `jax.lax.scan`.

All three transforms step a single-step `Node` along leading sequence axis 0
with shared parameters and per-step PRNG splitting, differing in how state
enters and leaves the run:

- `scan` keeps `state` external (`(state, xs) -> (state, ys)`), supporting
  multi-chunk streaming and named `boundary=` partial state resets.
- `scanned` internalizes `state` (`xs -> ys`), priming fresh state from the
  first sequence element (or starting from a bound `PSNode`) and discarding the
  final carry.
- `carried` internalizes `state` (`xs -> final_psnode`), priming or consuming
  initial state and returning `step` bound to its final state as a `PSNode`
  while discarding per-step primary outputs.
"""

from __future__ import annotations

from nodejax.core.binding import split_aux
from nodejax.core.contract import Contract
from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.core.spec import add_axis, element_spec
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import bind, scan_inputs, scan_steps, transform
from nodejax.tree import tree_first


def _sequence_spec(inner: Contract, n: int | None = None):
    step = inner._def.calls.apply.input_spec
    if (
        step is None
        and n is not None
        and inner.apply_fields
        and not inner._apply_form.open
    ):
        step = inner._apply_form.declaration
    if step is None or (issubclass(type(step), Struct) and not step):
        return None
    return Struct(**{
        key: add_axis(value, n, fixed=n is not None)
        for key, value in step.__items__
    })


def _split_state_fields(input: Struct, fields: tuple[str, ...]) -> tuple[Struct, Struct]:
    state_input = Struct(**{name: input[name] for name in fields})
    sequence = Struct(**{name: value for name, value in input.__items__ if name not in fields})
    return state_input, sequence


def _internalized_form(inner: Contract, fields: tuple[str, ...],
                       n: int | None = None) -> dict:
    if not fields:
        return {'input_spec': _sequence_spec(inner, n)}
    return {'apply_fields': tuple(inner.apply_fields) + fields, 'input_spec': None}


def _check_length(name: str, n: int | None) -> None:
    if n is not None and (type(n) is not int or n < 1):
        raise TypeError(f'{name} n must be a positive int, got {n!r}')


def _state_at_first_step(inner: Contract, param, state_input, data, rng, *,
                         bundled: bool):
    """Primes fresh state from the concrete first element of `data`."""
    element = tree_first(scan_inputs(inner, data))
    if bundled:
        element = inner.intake(element)
    return inner.prime(param, state_input, element, rng)


def _element_initialize(contract, param, state_input, rng):
    current = contract.members.step
    sequence_spec = contract.input_spec
    element = (None if sequence_spec is None else
               element_spec(sequence_spec))
    current = (current if element is None else
               current._resolve_def(element, bundled=True).contract)
    return current.init(
        param, state_input, rng)


def _sequence_parameterize(member_name: str, fields: tuple[str, ...] = ()):
    def parameterize(contract, param_input, rng):
        current = getattr(contract.members, member_name)
        sequence_spec = contract.input_spec
        if sequence_spec is not None:
            _, sequence_spec = _split_state_fields(sequence_spec, fields)
            current = current._resolve_def(
                element_spec(sequence_spec), bundled=True).contract
        return current.param(param_input, rng)

    return parameterize


def _fresh_step_state(step: Contract, param, state_input, inputs, rng):
    child_rng = rng.child(step.init_takes_rng)
    return _state_at_first_step(
        step, param, state_input, inputs, child_rng, bundled=True)


def _run_start(contract: Contract, param, state_input, sequence, rng, state):
    current = contract.members.step
    if state is not None:
        return current._dense_state(state)
    if not current.cyclic:
        return ()
    return _fresh_step_state(current, param, state_input, sequence, rng)


def _run_takes_rng(inner: Contract, state) -> bool:
    if state is None:
        return inner.init_takes_rng or inner.apply_takes_rng
    return inner.apply_takes_rng


def _check_claim(inner: Contract, boundary: str | None) -> None:
    if boundary is None:
        return
    declared = inner.boundary_names
    if boundary in declared:
        return
    known = (f'; this subtree declares {sorted(declared)}' if declared
             else '; nothing beneath it declares any boundary')
    raise TypeError(
        f"scan({inner.name}): boundary={boundary!r} fires nothing{known}. "
        'Name the boundary the nodes name, or drop the argument: carrying is '
        'what happens either way, so the claim is doing nothing for you.')


@transform(preserves='param,state')
def scan(step: Node, record: bool = False,
         boundary: str | None = None,
         n: int | None = None) -> Node | PNode:
    """Steps `step` along leading sequence axis 0 while threading `state` externally.

    Unlike `scanned` and `carried`, `scan` keeps `state` on the outer contract
    so carry persists across chunked calls. Supplying `boundary` reinitializes
    only descendant states marked with `state_reinit(..., boundary)` at the
    start of each call; `record=True` sows the per-step state trajectory under
    `Aux(state=...)`; and `n` enforces a fixed sequence length.
    """
    _check_length('scan', n)
    _check_claim(step.contract, boundary)

    if not step.cyclic:
        # Acyclic steps carry `()` across the scan and expose a stateless sequence map.
        def map_fn(contract, param, input, rng):
            current = contract.members.step
            input = scan_inputs(current, input, n)
            _, outputs = scan_steps(
                current, param, (), input, rng, record=record, length=n)
            return outputs

        return Wrapper(step=step).roles(
            name=f'scan({step.name})',
            param=_sequence_parameterize('step'),
            init=False,
            apply=map_fn,
            input_spec=_sequence_spec(step.contract, n),
            apply_takes_rng=step.contract.apply_takes_rng,
        )

    def prime_fn(contract, param, state_input, input, rng):
        current = contract.members.step
        input = scan_inputs(current, input, n)
        return _state_at_first_step(
            current, param, state_input, input, rng, bundled=False)

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.step
        input = scan_inputs(current, input, n)
        start = state
        if boundary is not None:
            start = current.merge_boundary(
                state,
                _state_at_first_step(
                    current,
                    param,
                    Struct(),
                    input,
                    rng.child(current.init_takes_rng),
                    bundled=True,
                ),
                boundary)
        return scan_steps(
            current,
            param,
            start,
            input,
            rng,
            record=record,
            length=n,
        )

    sequence_spec = _sequence_spec(step.contract, n)
    return Wrapper(step=step).roles(
        name=f'scan({step.name})',
        param=_sequence_parameterize('step'),
        init=_element_initialize,
        prime=prime_fn,
        apply=apply_fn,
        input_spec=sequence_spec,
        apply_takes_rng=(
            (boundary is not None and step.contract.init_takes_rng)
            or step.contract.apply_takes_rng),
    )


@transform(preserves='param', internalizes='state')
def scanned(step: Node, record: bool = False,
            n: int | None = None) -> Node | PNode:
    """Runs `step` across a sequence from fresh or bound state and returns stacked outputs.

    Unlike `scan`, `scanned` exposes a stateless outer Node whose call accepts
    any unbatched state-input fields alongside the sequence, primes initial
    state from the first element (unless `step` was passed as a `PSNode`), and
    discards the final carry. Setting `record=True` sows the state trajectory
    under `Aux(state=...)`, and `n` enforces a fixed sequence length.
    """
    _check_length('scanned', n)
    state = step.state if step.state_bound else None
    step = step.node
    fields = () if state is not None else step.contract.state_input_fields

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, sequence = _split_state_fields(input, fields)
        sequence = scan_inputs(current, sequence, n)
        initial = _run_start(contract, param, state_input, sequence, rng, state)
        _, outputs = scan_steps(
            current, param, initial, sequence, rng, record=record, length=n)
        return outputs

    return Wrapper(step=step).roles(
        name=f'scanned({step.name})',
        param=_sequence_parameterize('step', fields),
        init=False,
        apply=apply_fn,
        **_internalized_form(step.contract, fields, n),
        apply_takes_rng=_run_takes_rng(step.contract, state),
    )


@transform(preserves='param', internalizes='state')
def carried(step: Node, n: int | None = None) -> Node | PNode:
    """Runs cyclic `step` across a sequence and returns a `PSNode` bound to the final state.

    Unlike `scanned`, which returns stacked per-step outputs and discards the
    final carry, `carried` discards per-step primary outputs (retaining stacked
    `Aux` values when emitted) and returns `step` bound to its terminal state so
    subsequent calls or member inspections read the warmed-up carry directly.
    """
    if not step.cyclic:
        raise TypeError(f'carried requires a cyclic node, got {step!r}')
    _check_length('carried', n)
    state = step.state if step.state_bound else None
    step = step.node
    fields = () if state is not None else step.contract.state_input_fields

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, sequence = _split_state_fields(input, fields)
        sequence = scan_inputs(current, sequence, n)
        initial = _run_start(contract, param, state_input, sequence, rng, state)
        final, outputs = scan_steps(
            current, param, initial, sequence, rng, length=n)
        _, aux = split_aux(outputs)
        done = bind(current, param, state=final)
        return done if aux is None else (done, aux)

    return Wrapper(step=step).roles(
        name=f'carried({step.name})',
        param=_sequence_parameterize('step', fields),
        init=False,
        apply=apply_fn,
        **_internalized_form(step.contract, fields, n),
        apply_takes_rng=_run_takes_rng(step.contract, state),
    )
