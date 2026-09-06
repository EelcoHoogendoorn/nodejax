"""Supported building blocks for authoring NodeJAX transforms.

This module is deliberately opt-in. Regular users choose ready-made
transforms from :mod:`nodejax.transforms`; transform authors use this small
surface instead of importing framework-private helpers.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.core.ambient import _NO_REPLAY_ARGUMENT, node
from nodejax.core.binding import Aux, AxisSpec, REQUIRED, split_aux
from nodejax.core.contract import Contract
from nodejax.core.node import Node, _is_node
from nodejax.core.pnode import PNode
from nodejax.core.psnode import PSNode
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.spec import add_axis, axis_count, element_spec
from nodejax.frozendict import frozendict
from nodejax.struct import Struct
from nodejax.tree import tree_first


_NO_STATE = object()


def bind(contract: Contract, param, *, state=_NO_STATE):
    """Construct a public bound view from a T3 Contract and value trees."""
    if type(contract) is not Contract:
        raise TypeError('transform.bind expects a Contract')
    if state is _NO_STATE:
        return PNode(contract._def, param)
    return PSNode(contract._def, param, state)


def _split_and_tile(tree, count: int):
    """Tile construction data; split an authored stored ``rng`` field."""
    if type(tree) is Struct:
        return Struct(**{
            name: (jax.random.split(value, count) if name == 'rng'
                   else _split_and_tile(value, count))
            for name, value in tree.__items__})
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf), (count,) + jnp.shape(leaf)),
        tree,
    )


def vmap_apply(inner: Contract, param, state, input, rng: MaybeKeyStream, *,
               param_axis: Any, state_axis: Any, input_axis: Any,
               axis_name: str, count: int | None = None) -> tuple[Any, Any]:
    """Apply one node over declared parameter, state, and input axes.

    An axis declaration for an absent role is ignored. Transform bodies state
    their mapping policy; the helper owns canonical empty-role handling.
    """
    param_axis = param_axis if inner.parametric else None
    state_axis = state_axis if inner.cyclic else None
    if param_axis is not None and count is not None:
        actual = axis_count(param)
        if actual != count:
            raise TypeError(
                f'{inner.name}: mapped parameter axis has {actual} rows; '
                f'expected {count}')
    size = count
    if inner.apply_takes_rng and size is None:
        source = (input if input_axis is not None else
                  param if param_axis is not None else state)
        size = axis_count(source)

    evidence = tree_first(input) if input_axis is not None else input
    contract = (inner if evidence is None else
                inner._resolve_def(evidence, bundled=True).contract)
    rngs, rng_axis = rng.axis(contract.apply_takes_rng, size)
    return jax.vmap(
        lambda p, s, x, child_rng: contract.apply(p, s, x, child_rng),
        in_axes=(param_axis, state_axis, input_axis, rng_axis),
        out_axes=(state_axis, 0),
        axis_name=axis_name,
        axis_size=count,
    )(param, state, input, rngs)


def vmap_init(inner: Contract, outer: Contract, rng: MaybeKeyStream,
              param, state_input: Struct, *,
              count: int, param_axis: int | None) -> Any:
    """Initialize one non-priming state per mapped member."""
    param_axis = param_axis if inner.parametric else None
    shape = outer.input_spec
    contract = (inner if shape is None else
                inner._resolve_def(shape, bundled=True).contract)
    rngs, rng_axis = rng.axis(contract.init_takes_rng, count)
    if param_axis is not None or rng_axis is not None:
        return jax.vmap(
            lambda member_param, child_rng: contract.init(
                member_param, state_input, child_rng),
            in_axes=(param_axis, rng_axis),
            axis_size=count,
        )(param, rngs)
    return _split_and_tile(
        contract.init(param, state_input, rngs), count)


def vmap_prime(inner: Contract, param, state_input: Struct, input,
               rng: MaybeKeyStream, *, count: int,
               param_axis: int | None, input_axis: int | None,
               state_axis: Any, axis_name: str | None = None) -> Any:
    """Prime one state per mapped member under an optional named axis."""
    param_axis = param_axis if inner.parametric else None
    state_axis = state_axis if inner.cyclic else None
    rngs, rng_axis = rng.axis(inner.init_takes_rng, count)
    if (param_axis is not None or input_axis is not None
            or rng_axis is not None or axis_name is not None):
        return jax.vmap(
            lambda member_param, member_input, child_rng: inner.prime(
                member_param, state_input, member_input, child_rng),
            in_axes=(param_axis, input_axis, rng_axis),
            out_axes=state_axis,
            axis_name=axis_name,
            axis_size=count,
        )(param, input, rngs)
    return _split_and_tile(
        inner.prime(param, state_input, input, rngs), count)


def vmap_param(inner: Contract, outer: Contract, rng: MaybeKeyStream,
               param_input: Struct, *, count: int) -> Any:
    """Construct ``count`` independent parameter rows."""
    shape = outer.input_spec
    contract = (inner if shape is None else
                inner._resolve_def(shape, bundled=True).contract)
    tiled = _split_and_tile(param_input, count)
    rngs, rng_axis = rng.axis(contract.param_takes_rng, count)
    return jax.vmap(
        lambda bundle, child_rng: contract.param(bundle, child_rng),
        in_axes=(0, rng_axis),
        axis_size=count,
    )(tiled, rngs)


def _record_state(output, state, name):
    clean, aux = split_aux(output)
    fields = dict(aux.__items__) if aux is not None else {}
    if 'state' in fields:
        raise TypeError(
            f"scan({name}, record=True): the node already sows 'state'; "
            'rename that field or disable recording')
    return clean, Aux(**fields, state=state)


def scan_inputs(
    step: Contract,
    input: Struct,
    length: int | None = None,
) -> Struct:
    """Validate that every input field carries one common sequence axis.

    A step with no input fields scans for a declared ``length``: a system
    ticking on its own."""
    leaves = jax.tree.leaves(input)
    if not leaves:
        if length is None:
            raise TypeError(
                f'scan({step.name}) needs at least one input field with a '
                'leading sequence axis, or a declared n')
        return input
    if any(not jnp.shape(leaf) for leaf in leaves):
        raise TypeError(
            f'scan({step.name}) received a scalar where a sequence axis '
            'is required')
    count = jnp.shape(leaves[0])[0]
    if any(jnp.shape(leaf)[0] != count for leaf in leaves[1:]):
        raise TypeError(f'scan({step.name}) input axes have unequal lengths')
    if length is not None and count != length:
        raise TypeError(
            f'scan({step.name}) input axis has {count} steps; '
            f'expected n={length}')

    return input


def scan_steps(step: Contract, param, state, inputs, rng: MaybeKeyStream, *,
               record: bool = False, length: int | None = None):
    """Run one stateful node over a sequence with per-step RNG streams."""
    inputs = scan_inputs(step, inputs, length)
    leaves = jax.tree.leaves(inputs)
    count = leaves[0].shape[0] if leaves else length
    rngs, _ = rng.axis(step.apply_takes_rng, count)

    def body(carry, item):
        element, child_rng = item
        successor, output = step.apply(
            param, carry, element, child_rng)
        emitted = (_record_state(output, successor, step.name)
                   if record else output)
        return successor, emitted

    return jax.lax.scan(body, state, (inputs, rngs), length=length)


def _preserved_roles(preserves) -> tuple[str, ...]:
    if type(preserves) is str:
        roles = tuple(part.strip() for part in preserves.split(',')
                      if part.strip())
    else:
        roles = tuple(preserves)
    if roles not in ((), ('param',), ('param', 'state')):
        raise TypeError(
            "preserves must be (), 'param', or 'param,state'")
    return roles


def transform(builder: Callable | None = None, *, preserves=(),
              internalizes: str | None = None) -> Callable:
    """Record a node transform and lift the bindings it preserves.

    ``preserves=()`` accepts only an unbound :class:`Node` because the
    transform changes parameter or state layout. ``'param'`` also accepts a
    :class:`PNode` and reattaches its parameters. ``'param,state'`` does the
    same for a :class:`PSNode` and its state. ``internalizes='state'`` also
    accepts a :class:`PSNode` whose state the builder consumes as the start
    of the run it owns: the builder receives the bound view itself and only
    the parameters are reattached to the result. Otherwise the decorated
    builder receives an unbound Node.
    """
    roles = _preserved_roles(preserves)
    if internalizes not in (None, 'state'):
        raise TypeError("internalizes must be None or 'state'")

    def decorate(fn: Callable) -> Callable:
        public_signature = inspect.signature(fn)

        @wraps(fn)
        def lifted(inner, *args, **kwargs):
            if not _is_node(inner):
                raise TypeError('a transform expects a Node, PNode, or PSNode')
            internalized_state = kwargs.pop(
                '_internalized_state', _NO_REPLAY_ARGUMENT)
            state_recorded = internalized_state is not _NO_REPLAY_ARGUMENT
            if state_recorded:
                inner = PSNode(
                    inner._def,
                    internalized_state['param'],
                    internalized_state['state'],
                )
            if (internalizes == 'state' and inner.state_bound
                    and not state_recorded):
                product = registered(
                    inner.node,
                    *args,
                    _internalized_state=frozendict(
                        param=inner.param, state=inner.state),
                    **kwargs,
                )
                construction = product._def.construction
                replay_values = dict(
                    construction.replay_arguments.__items__)
                replay_values['_internalized_state'] = (
                    construction.arguments['_internalized_state'])
                arguments = Struct(**{
                    name: value
                    for name, value in construction.arguments.__items__
                    if name != '_internalized_state'
                })
                definition = product._def.copy(construction=construction.copy(
                    arguments=arguments,
                    replay_arguments=Struct(**replay_values),
                ))
                return product._with_definition(definition)
            source = inner
            if internalizes == 'state' and inner.state_bound:
                product = fn(inner, *args, **kwargs)
                source = inner.pnode
            else:
                product = fn(inner.node, *args, **kwargs)
            if not _is_node(product):
                raise TypeError(
                    f"transform '{fn.__name__}' did not return a Node")
            return source._transfer_bindings(
                product, roles, strict=True, operation='this transform')

        if internalizes == 'state':
            if '_internalized_state' in public_signature.parameters:
                raise TypeError(
                    "an internalizing transform reserves '_internalized_state'")
            parameters = list(public_signature.parameters.values())
            insertion = next(
                (index for index, parameter in enumerate(parameters)
                 if parameter.kind is inspect.Parameter.VAR_KEYWORD),
                len(parameters),
            )
            parameters.insert(insertion, inspect.Parameter(
                '_internalized_state',
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=_NO_REPLAY_ARGUMENT,
            ))
            lifted.__signature__ = public_signature.replace(
                parameters=parameters)

        registered = node(lifted)
        registered.__signature__ = public_signature
        return registered

    return decorate if builder is None else decorate(builder)


__all__ = [
    'transform', 'bind', 'Contract', 'MaybeKeyStream',
    'AxisSpec', 'add_axis', 'element_spec', 'axis_count',
    'vmap_param', 'vmap_init', 'vmap_prime', 'vmap_apply',
    'scan_inputs', 'scan_steps',
]
