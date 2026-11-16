"""T3 decorator and `vmap`/`scan` contract helpers for authoring NodeJAX transforms.

`@transform` wraps a `Node -> Node` builder so construction is recorded for
`specialize` and bound `PNode` or `PSNode` views reattach preserved trees
according to `preserves` and `internalizes`. Companion helpers (`vmap_param`,
`vmap_init`, `vmap_prime`, `vmap_apply`, `scan_inputs`, `scan_steps`) map or
scan `Contract` roles while handling empty stateless/non-parametric trees and
`MaybeKeyStream` splitting uniformly.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable

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
from nodejax.core.types import PyTree
from nodejax.frozendict import frozendict
from nodejax.struct import Struct
from nodejax.tree import tree_first


_NO_STATE = object()


def bind(contract: Contract, param, *, state=_NO_STATE) -> PNode | PSNode:
    """Attaches `param` (and optional `state`) to `contract`, returning a `PNode` or `PSNode`."""
    if type(contract) is not Contract:
        raise TypeError('transform.bind expects a Contract')
    if state is _NO_STATE:
        return PNode(contract._def, param)
    return PSNode(contract._def, param, state)


def _split_and_tile(tree: PyTree, count: int) -> PyTree:
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf), (count,) + jnp.shape(leaf)),
        tree,
    )


def vmap_apply(inner: Contract, param, state, input, rng: MaybeKeyStream, *,
               param_axis: PyTree, state_axis: PyTree, input_axis: PyTree,
               axis_name: str,
               count: int | None = None) -> tuple[PyTree, PyTree]:
    """Evaluates `inner.apply` under `jax.vmap` across declared parameter, state, and input axes.

    Axis declarations for absent parameter or state roles collapse to `None`
    automatically, and stochastic `apply` roles split `rng` across the mapped
    extent.
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
              count: int, param_axis: int | None) -> PyTree:
    """Initializes `count` mapped member states via `jax.vmap` or tiling when deterministic."""
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
               state_axis: PyTree, axis_name: str | None = None) -> PyTree:
    """Primes `count` mapped member states against `input` under an optional named `vmap` axis."""
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
               param_input: Struct, *, count: int) -> PyTree:
    """Initializes `count` independent parameter rows along leading axis 0."""
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
    """Checks that all leaves in `input` share a common leading sequence axis matching `length`.

    When `input` has no array leaves, `length` must be supplied explicitly so
    an autonomous zero-input system knows how many steps to run.
    """
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
               record: bool = False,
               length: int | None = None) -> tuple[PyTree, PyTree]:
    """Scans `step` across `inputs` via `jax.lax.scan` with per-step RNG splitting."""
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
    """Decorates a `Node` transform, records construction for `specialize`, and lifts preserved bindings.

    With `preserves=()`, the transform alters parameter or state layout and
    accepts only an unbound `Node`. Setting `preserves='param'` also accepts a
    `PNode` and reattaches its parameter tree, while `preserves='param,state'`
    additionally reattaches state on a `PSNode`. Setting `internalizes='state'`
    accepts a `PSNode`, passes the bound view into `builder` as the initial
    carry of the transformed run, and reattaches only `param` on the returned
    view.
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
