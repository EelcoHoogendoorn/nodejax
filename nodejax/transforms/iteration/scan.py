from __future__ import annotations

from nodejax.core.binding import _has_rng_deep, split_aux
from nodejax.core.contract import Contract
from nodejax.core.definition import Captures
from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.core.spec import add_axis, element_spec
from nodejax.frozendict import frozendict
from nodejax.struct import Struct
from nodejax.tree import tree_first
from nodejax.transforms.transform import bind, scan_inputs, scan_steps, transform
from nodejax.core.wrapper import Wrapper


def _sequence_spec(inner: Contract, n: int | None = None):
    """Lift a step input specification over one sequence axis."""
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
    """One internalized call as its state-input bundle and its sequence bundle."""
    state_input = Struct(**{name: input[name] for name in fields})
    sequence = Struct(**{name: value for name, value in input.__items__ if name not in fields})
    return state_input, sequence


def _internalized_form(inner: Contract, fields: tuple[str, ...],
                       n: int | None = None) -> dict:
    """The call form of an internalized run: the step's sequence fields, and its
    required state-input fields beside them, once and unbatched over time."""
    if not fields:
        return {'input_spec': _sequence_spec(inner, n)}
    return {'apply_fields': tuple(inner.apply_fields) + fields, 'input_spec': None}


def _check_length(name: str, n: int | None) -> None:
    if n is not None and (type(n) is not int or n < 1):
        raise TypeError(f'{name} n must be a positive int, got {n!r}')


def _state_at_first_step(inner: Contract, param, state_input, data, rng, *,
                         bundled: bool):
    """Build fresh state from the first element of a sequence.

    `scan` needs what a restart would produce at a claimed boundary;
    internalized runs need the same fresh carry as their start. The element is
    a real value, so a node that primes gets data and not a shape, and the node
    is bound to it so a node that only reads its shape is served too.
    """
    element = tree_first(scan_inputs(inner, data))
    if bundled:
        element = inner.intake(element)
    return inner.prime(param, state_input, element, rng)


def _element_initialize(contract, param, state_input, rng):
    """Build the step's state from one element's shape alone."""
    current = contract.members.step
    sequence_spec = contract.input_spec
    element = (None if sequence_spec is None else
               element_spec(sequence_spec))
    current = (current if element is None else
               current._resolve_def(element, bundled=True).contract)
    return current.init(
        param, state_input, rng)


def _sequence_parameterize(member_name: str, fields: tuple[str, ...] = ()):
    """``fields`` are the state-input fields this transform's own call takes
    beside the sequence; they carry no sequence axis."""
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
    """Initialize an internalized run from its state inputs and its first
    sequence element."""
    child_rng = rng.child(step.init_takes_rng)
    return _state_at_first_step(
        step, param, state_input, inputs, child_rng, bundled=True)


def _run_start(contract: Contract, param, state_input, sequence, rng):
    """The carry an internalized run starts from: the state the run was
    given, held as its step's capture, else fresh state from the state
    inputs and the first sequence element."""
    current = contract.members.step
    captured = contract._def.captures.state
    if 'step' in captured:
        return contract.member_init('step', param, captured['step'], rng)
    if not current.cyclic:
        return ()
    return _fresh_step_state(current, param, state_input, sequence, rng)


def _from_state(run: Node, state) -> Node:
    """Hold the state a run starts from as its step's capture, so it is
    rekeyed and described like any bound member value."""
    if state is None:
        return run
    definition = run._def
    return run._with_definition(definition.copy(captures=Captures(
        param=definition.captures.param, state=frozendict(step=state))))


def _run_takes_rng(inner: Contract, state) -> bool:
    if state is None:
        return inner.init_takes_rng or inner.apply_takes_rng
    return inner.apply_takes_rng or _has_rng_deep(state)


def _check_claim(inner: Contract, boundary):
    """Reject a boundary name that no node in the step declares."""
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
    """Run ``step`` over a sequence while keeping its state external.

    The supplied state becomes the initial carry and the returned state can be
    passed to another call, making this form suitable for chunks or open-ended
    streams. ``boundary`` may reinitialize matching state slots at the start
    of each call. ``record=True`` adds the state trajectory to the auxiliary
    output. ``n`` declares and enforces a fixed sequence length; without it,
    the sequence length may change between calls.
    """
    _check_length('scan', n)
    _check_claim(step.contract, boundary)

    if not step.cyclic:
        # An acyclic step has unit carry, so scanning it is its ordinary
        # sequence map; the boundary claim above already rejects a
        # boundary= that nothing beneath declares.
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
def scanned(step: Node, record: bool = False, n: int | None = None, *,
            state=None) -> Node | PNode:
    """Run ``step`` over a sequence with fresh state on every call.

    A state-bound step runs from its bound state, which ``state`` gives
    directly for an unbound step. Otherwise the state is built from the
    step's state-input fields, which the call takes beside the sequence
    fields, once and unbatched over time, and from the first sequence
    element. The per-step outputs are returned and the final state is
    discarded. ``record=True`` adds the state trajectory to the auxiliary
    output without changing the ordinary output. ``n`` declares and
    enforces a fixed sequence length; without it, the sequence length may
    change between calls. An acyclic step has unit state, so this is its
    ordinary sequence map.
    """
    _check_length('scanned', n)
    fields = () if state is not None else step.contract.state_input_fields

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, sequence = _split_state_fields(input, fields)
        sequence = scan_inputs(current, sequence, n)
        initial = _run_start(contract, param, state_input, sequence, rng)
        _, outputs = scan_steps(
            current, param, initial, sequence, rng, record=record, length=n)
        return outputs

    run = Wrapper(step=step).roles(
        name=f'scanned({step.name})',
        param=_sequence_parameterize('step', fields),
        init=False,
        apply=apply_fn,
        **_internalized_form(step.contract, fields, n),
        apply_takes_rng=_run_takes_rng(step.contract, state),
    )
    return _from_state(run, state)


@transform(preserves='param', internalizes='state')
def carried(step: Node, n: int | None = None, *, state=None) -> Node | PNode:
    """Run ``step`` from fresh state and return it bound to its final state.

    A state-bound step runs from its bound state, which ``state`` gives
    directly for an unbound step. Otherwise the state is built from the
    step's state-input fields, which the call takes beside the sequence
    fields, and from the first sequence element. Per-step outputs are
    discarded. Any auxiliary values emitted by the step remain available
    alongside the bound step, whose members and methods read the final
    state. ``n`` declares and enforces a fixed sequence length.
    """
    if not step.cyclic:
        raise TypeError(f'carried requires a cyclic node, got {step!r}')
    _check_length('carried', n)
    fields = () if state is not None else step.contract.state_input_fields

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        state_input, sequence = _split_state_fields(input, fields)
        sequence = scan_inputs(current, sequence, n)
        initial = _run_start(contract, param, state_input, sequence, rng)
        final, outputs = scan_steps(
            current, param, initial, sequence, rng, length=n)
        _, aux = split_aux(outputs)
        done = bind(current, param, state=final)
        return done if aux is None else (done, aux)

    run = Wrapper(step=step).roles(
        name=f'carried({step.name})',
        param=_sequence_parameterize('step', fields),
        init=False,
        apply=apply_fn,
        **_internalized_form(step.contract, fields, n),
        apply_takes_rng=_run_takes_rng(step.contract, state),
    )
    return _from_state(run, state)
