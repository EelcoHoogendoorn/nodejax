"""Supported building blocks for authoring NodeJAX transforms.

This module is deliberately opt-in.  Regular users choose ready-made
transforms from :mod:`nodejax.transforms`; transform authors use this small
surface instead of importing framework-private helpers.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.ambient import node
from nodejax.binding import Aux, AxisSpec, REQUIRED, split_aux
from nodejax.contract import Contract
from nodejax.node import Node, _is_node
from nodejax.pnode import PNode
from nodejax.psnode import PSNode
from nodejax.rng import MaybeKeyStream
from nodejax.spec import add_axis, axis_count, element_spec, tree_first
from nodejax.struct import Struct


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
               state_axis: Any) -> Any:
    """Prime one state per mapped member from real input values."""
    param_axis = param_axis if inner.parametric else None
    state_axis = state_axis if inner.cyclic else None
    rngs, rng_axis = rng.axis(inner.init_takes_rng, count)
    if param_axis is not None or input_axis is not None or rng_axis is not None:
        return jax.vmap(
            lambda member_param, member_input, child_rng: inner.prime(
                member_param, state_input, member_input, child_rng),
            in_axes=(param_axis, input_axis, rng_axis),
            out_axes=state_axis,
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


def scan_inputs(step: Contract, input: Struct) -> Struct:
    """Validate that every input field carries one common sequence axis."""
    leaves = jax.tree.leaves(input)
    if not leaves:
        raise TypeError(
            f'scan({step.name}) needs at least one input field with a '
            'leading sequence axis')
    if any(not jnp.shape(leaf) for leaf in leaves):
        raise TypeError(
            f'scan({step.name}) received a scalar where a sequence axis '
            'is required')
    count = jnp.shape(leaves[0])[0]
    if any(jnp.shape(leaf)[0] != count for leaf in leaves[1:]):
        raise TypeError(f'scan({step.name}) input axes have unequal lengths')

    return input


def scan_steps(step: Contract, param, state, inputs, rng: MaybeKeyStream, *,
               record: bool = False):
    """Run one stateful node over a sequence with per-step RNG streams."""
    inputs = scan_inputs(step, inputs)
    leaves = jax.tree.leaves(inputs)
    if step.apply_takes_rng and not leaves:
        raise TypeError(
            f'scan({step.name}): a stochastic step needs a non-empty input '
            'pytree to determine the sequence length')
    count = leaves[0].shape[0] if leaves else None
    rngs, _ = rng.axis(step.apply_takes_rng, count)

    def body(carry, item):
        element, child_rng = item
        successor, output = step.apply(
            param, carry, element, child_rng)
        emitted = (_record_state(output, successor, step.name)
                   if record else output)
        return successor, emitted

    return jax.lax.scan(body, state, (inputs, rngs))


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


def transform(builder: Callable | None = None, *, preserves=()) -> Callable:
    """Record a node transform and lift the bindings it preserves.

    ``preserves=()`` accepts only an unbound :class:`Node` because the
    transform changes parameter or state layout. ``'param'`` also accepts a
    :class:`PNode` and reattaches its parameters. ``'param,state'`` does the
    same for a :class:`PSNode` and its state.  The decorated builder itself
    always receives an unbound Node.
    """
    roles = _preserved_roles(preserves)

    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def lifted(inner, *args, **kwargs):
            def build(current):
                product = fn(current, *args, **kwargs)
                if not _is_node(product):
                    raise TypeError(
                        f"transform '{fn.__name__}' did not return a Node")
                return product

            if not _is_node(inner):
                raise TypeError('a transform expects a Node, PNode, or PSNode')
            return inner._transfer_bindings(
                build(inner.node), roles, strict=True,
                operation='this transform')

        return node(lifted)

    return decorate if builder is None else decorate(builder)


__all__ = [
    'transform', 'bind', 'Contract', 'MaybeKeyStream',
    'AxisSpec', 'add_axis', 'element_spec', 'axis_count',
    'vmap_param', 'vmap_init', 'vmap_prime', 'vmap_apply',
    'scan_inputs', 'scan_steps',
]
