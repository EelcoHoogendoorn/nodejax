from __future__ import annotations

from nodejax.binding import split_aux
from nodejax.contract import Contract
from nodejax.node import Node
from nodejax.pnode import PNode
from nodejax.spec import add_axis, element_spec, tree_first
from nodejax.struct import Struct
from nodejax.transform import scan_inputs, scan_steps, transform
from nodejax.wrapper import Wrapper


def _sequence_spec(inner: Contract):
    """Lift a step input specification over a variable-length time axis."""
    step = inner.input_spec
    if step is None:
        return None
    return Struct(**{
        k: add_axis(v, fixed=False) for k, v in step.__items__})


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


def _prime_from_first_element(contract, param, state_input, input, rng):
    """Prime the step's state from the sequence's first real element."""
    current = contract.members.step
    return _state_at_first_step(
        current, param, state_input, input, rng, bundled=False)


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


def _sequence_parameterize(member_name: str):
    def parameterize(contract, param_input, rng):
        current = getattr(contract.members, member_name)
        sequence_spec = contract.input_spec
        if sequence_spec is not None:
            current = current._resolve_def(
                element_spec(sequence_spec), bundled=True).contract
        return current.param(param_input, rng)

    return parameterize


def _fresh_step_state(step: Contract, param, inputs, rng):
    """Initialize an internalized run from its first sequence element."""
    child_rng = rng.child(step.init_takes_rng)
    return _state_at_first_step(
        step, param, Struct(), inputs, child_rng, bundled=True)


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
         boundary: str | None = None) -> Node | PNode:
    """Run ``step`` over a sequence while keeping its state external.

    The supplied state becomes the initial carry and the returned state can be
    passed to another call, making this form suitable for chunks or open-ended
    streams. ``boundary`` may reinitialize matching state slots at the start
    of each call. ``record=True`` adds the state trajectory to the auxiliary
    output.
    """
    if not step.cyclic:
        raise TypeError(f'scan requires a cyclic node, got {step!r}')
    _check_claim(step.contract, boundary)

    def apply_fn(contract, param, state, input, rng):
        current = contract.members.step
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
            current, param, start, input, rng, record=record)

    sequence_spec = _sequence_spec(step.contract)
    return Wrapper(step=step).roles(
        name=f'scan({step.name})',
        param=_sequence_parameterize('step'),
        init=_element_initialize,
        prime=_prime_from_first_element,
        apply=apply_fn,
        input_spec=sequence_spec,
        apply_takes_rng=(
            (boundary is not None and step.contract.init_takes_rng)
            or step.contract.apply_takes_rng),
    )


@transform(preserves='param')
def scanned(step: Node, record: bool = False) -> Node | PNode:
    """Run ``step`` over a sequence with fresh state on every call.

    The first sequence element initializes the state, the per-step outputs are
    returned, and the final state is discarded. ``record=True`` adds the state
    trajectory to the auxiliary output without changing the ordinary output.
    """
    if not step.cyclic:
        raise TypeError(f'scanned requires a cyclic node, got {step!r}')

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        initial = _fresh_step_state(current, param, input, rng)
        _, outputs = scan_steps(
            current, param, initial, input, rng, record=record)
        return outputs

    return Wrapper(step=step).roles(
        name=f'scanned({step.name})',
        param=_sequence_parameterize('step'),
        init=False,
        apply=apply_fn,
        input_spec=_sequence_spec(step.contract),
        apply_takes_rng=(step.contract.init_takes_rng
                         or step.contract.apply_takes_rng),
    )


@transform(preserves='param')
def carried(step: Node) -> Node | PNode:
    """Run ``step`` from fresh state and return its final state.

    Per-step outputs are discarded. Any auxiliary values emitted by the step
    remain available alongside the final state.
    """
    if not step.cyclic:
        raise TypeError(f'carried requires a cyclic node, got {step!r}')

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        initial = _fresh_step_state(current, param, input, rng)
        final, outputs = scan_steps(
            current, param, initial, input, rng)
        _, aux = split_aux(outputs)
        return final if aux is None else (final, aux)

    return Wrapper(step=step).roles(
        name=f'carried({step.name})',
        param=_sequence_parameterize('step'),
        init=False,
        apply=apply_fn,
        input_spec=_sequence_spec(step.contract),
        apply_takes_rng=(step.contract.init_takes_rng
                         or step.contract.apply_takes_rng),
    )
