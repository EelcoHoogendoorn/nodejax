"""Shared lifting helpers for the node transforms.

KNOWN GAP — spec propagation is partial. The mapped transforms below
resolve the inner def from the outer carry at param and init time, but
an outer with_input is not rewritten inward through wrapper transforms
(scan, ttt): a shape-reading constructor inside such a tower cannot be
resolved from outside, and the shape must be stated at construction
instead. Closing this means each transform knowing how to transform a
spec (strip the mapped axis, take the stream element, unwrap the
sample bundle), the input-side mirror of what the transforms already
do to params and state."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import Param
from nodejax.core import (Node, NodeDef, Wrapper, _input_or_none, _resolve,
                            _trivial_param_fn, _split_rng, _with_rng)


_UNBOUND: Any = object()  # sentinel: _split saw a NodeDef, not a bound Node
_KEEP: Any = object()     # sentinel: _transform_def preserves default field value


def _mapped_apply_fn(inner: NodeDef, param: Any, state: Any, input: Any, *,
                     param_axis: Any, state_axis: Any, input_axis: Any,
                     axis_name: str, count: int | None = None) -> tuple[Any, Any]:
    """Uniform vmap execution over param, state, and input axes, automatically
    splitting apply-side rng keys when the inner node consumes rng.

    state_axis serves as out_axes for the returned state as well as in_axes
    for the supplied one: a member mapped in is mapped out, and a member
    broadcast in (a node holding one state across the whole axis) comes back
    unmapped. jax enforces the second case, so a node that declares a shared
    state but computes a per-element one fails at the vmap rather than
    silently keeping one element's copy."""
    if inner.apply_takes_rng:
        N = count if count is not None else (
            jax.tree.leaves(input)[0].shape[0] if input_axis == 0 else
            jax.tree.leaves(param)[0].shape[0]
        )
        keys, data = _split_rng(input if input_axis == 0 else Struct(rng=input.rng), N)
        clean_input = input.without('rng') if input_axis == 0 else input
        if input_axis == 0:
            return jax.vmap(
                lambda p_, s_, i_, k_: inner.apply_fn(p_, s_, _with_rng(i_, k_)),
                in_axes=(param_axis, state_axis, 0, 0),
                out_axes=(state_axis, 0), axis_name=axis_name
            )(param, state, clean_input, keys)
        return jax.vmap(
            lambda p_, s_, k_: inner.apply_fn(p_, s_, _with_rng(clean_input, k_)),
            in_axes=(param_axis, state_axis, 0),
                out_axes=(state_axis, 0), axis_name=axis_name
        )(param, state, keys)

    if input_axis is None:
        return jax.vmap(
            lambda p_, s_: inner.apply_fn(p_, s_, input),
            in_axes=(param_axis, state_axis),
                out_axes=(state_axis, 0), axis_name=axis_name
        )(param, state)

    return jax.vmap(
        lambda p_, s_, i_: inner.apply_fn(p_, s_, i_),
        in_axes=(param_axis, state_axis, input_axis),
            out_axes=(state_axis, 0), axis_name=axis_name
    )(param, state, input)


def _scanned_apply_fn(inner: NodeDef, param: Any, state: Any, input: Any, *,
                   stacked_param: bool, length: int | None = None) -> tuple[Any, Any]:
    """Uniform lax.scan execution over a sequential axis: position k's output
    is position k+1's input, and the per-position states stack.

    `stacked_param` says whether params carry that axis, one row per position
    (stack's layers), or one set threads every position (repeat's tied
    weights). Apply-side rng splits per position, so no two positions share a
    draw. Aux rides the output as it does under vmap: split from the carry
    each step, stacked over the axis, re-emitted as the (output, aux) pair."""
    from nodejax.core import split_aux

    n = length if length is not None else jax.tree.leaves(param)[0].shape[0]
    if inner.apply_takes_rng:
        keys, first = _split_rng(input, n)
    else:
        keys, first = None, input

    # xs carries whatever varies per position; unpack names it back
    if stacked_param and keys is not None:
        scanned, unpack = (param, state, keys), lambda x: (x[0], x[1], x[2])
    elif stacked_param:
        scanned, unpack = (param, state), lambda x: (x[0], x[1], None)
    elif keys is not None:
        scanned, unpack = (state, keys), lambda x: (param, x[0], x[1])
    else:
        scanned, unpack = state, lambda x: (param, x, None)

    def step(carry, xs):
        p, s, key = unpack(xs)
        new_state, out = inner.apply_fn(p, s, carry if key is None else _with_rng(carry, key))
        clean_out, aux = split_aux(out)
        return clean_out, (new_state, aux)

    out, (new_states, auxs) = jax.lax.scan(step, first, scanned, length=n)
    return new_states, (out, auxs) if auxs is not None else out


def _transform_def(node_def: NodeDef, *,
                   name: str,
                   param_fn: Callable | None = None,
                   init_fn: Callable | None = None,
                   apply_fn: Callable | None = None,
                   parametric: bool | None = None,
                   cyclic: bool | None = None,
                   apply_input_spec: Any = _KEEP,
                   init_requires_input: bool | None = None,
                   init_reads_shape: bool | None = None,
                   param_reads_shape: bool | None = None,
                   param_input_spec: Any = _KEEP,
                   state_input_spec: Any = _KEEP,
                   tags: frozenset[str] | None = None,
                   rebuild: Callable[[NodeDef], NodeDef] | None = None) -> NodeDef:
    """Construct a transformed NodeDef from an existing one, preserving
    contract metadata while avoiding Composite/Serial subclass leakage."""
    p_fn = node_def._param_impl if param_fn is None else param_fn
    i_fn = node_def._init_impl if init_fn is None else init_fn
    a_fn = node_def._apply_impl if apply_fn is None else apply_fn
    p_flag = node_def.parametric if parametric is None else parametric
    c_flag = node_def.cyclic if cyclic is None else cyclic

    a_spec = node_def.apply_input_spec if apply_input_spec is _KEEP else apply_input_spec
    p_spec = (node_def.param_input_spec if p_flag else None) if param_input_spec is _KEEP else param_input_spec
    s_spec = (node_def.state_input_spec if c_flag else None) if state_input_spec is _KEEP else state_input_spec

    return Wrapper(
        inner=node_def,
        name=name,
        param_fn=p_fn,
        init_fn=i_fn,
        apply_fn=a_fn,
        parametric=p_flag,
        cyclic=c_flag,
        apply_input_spec=a_spec,
        init_requires_input=node_def.init_requires_input if init_requires_input is None else init_requires_input,
        param_reads_shape=node_def.param_reads_shape if param_reads_shape is None else param_reads_shape,
        init_reads_shape=node_def.init_reads_shape if init_reads_shape is None else init_reads_shape,
        param_input_spec=p_spec,
        state_input_spec=s_spec,
        tags=node_def.tags if tags is None else tags,
        rebuild=rebuild,
    )


def _split(x: NodeDef | Node) -> tuple[NodeDef, Param]:
    """Accept a NodeDef or a bound Node; return (def, param-or-_UNBOUND)."""
    if x.bound:
        return x.ndef, x.param
    return x.ndef, _UNBOUND


def _rewrap(ndef: NodeDef, param: Param) -> NodeDef | Node:
    """Rebind a transformed def, if the input was bound and the transform
    preserves the meaning of param."""
    return ndef if param is _UNBOUND else Node(ndef, param)


def _over_bound(transform: Callable) -> Callable:
    """ANNOTATION: this transform preserves the meaning of param, and
    therefore lifts over bound nodes — a Node in, the transformed def
    rebound to the same param out; the transform body speaks defs only.
    A transform WITHOUT this annotation rewrites what param means
    (ensemble stacks it per member, stack per layer) and refuses bound
    input explicitly instead."""
    @wraps(transform)
    def wrapped(node, *args, **kwargs):
        nd, param = _split(node)
        return _rewrap(transform(nd, *args, **kwargs), param)
    return wrapped


def _tile_state(state: Any, n: int) -> Any:
    """Tile a state pytree along a new leading axis of size n. Reserved
    'rng' fields are SPLIT into n independent keys rather than broadcast —
    a copied key would give every element the same noise stream."""
    if isinstance(state, Struct):
        return Struct(**{k: (jax.random.split(v, n) if k == 'rng' else _tile_state(v, n))
                         for k, v in state.__items__})
    return jax.tree.map(lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (n,) + jnp.shape(leaf)), state)


def _mapped_init_fn(inner: NodeDef, n: int | None = None, *,
                    stacked: bool | None = None) -> Callable:
    """init for transforms whose STATE gains a leading member axis: one state
    per member. `stacked` says whether the params carry that axis too —
    ensemble members and stack layers do (each member built from its own
    param row); repeat positions and non-parametric members share one param,
    so the count comes from n. A boundary rng in the seed always splits per
    member — an independent stream each, never a copy; a deterministic seed
    tiles (with any reserved state.rng field split, see _tile_state). Non-cyclic
    nodes build their empty state from a single unstacked param slice."""
    if stacked is None:
        stacked = inner.parametric

    def init_fn(ndef, p, state_input=Struct(), input=None):
        carry = input if input is not None else _input_or_none(ndef)
        d = inner if carry is None else _resolve(inner, carry)
        if not inner.cyclic:
            p0 = jax.tree.map(lambda x: x[0], p) if (stacked and inner.parametric and p != ()) else p
            return d.build_state(p0, state_input, input=carry)

        seed = state_input.without('rng')
        if stacked:
            if 'rng' in state_input:
                keys = jax.random.split(state_input.rng, jax.tree.leaves(p)[0].shape[0])
                return jax.vmap(lambda p_, k: d.build_state(
                    p_, seed.replace(rng=k), input=carry))(p, keys)
            return jax.vmap(lambda p_: d.build_state(p_, seed, input=carry))(p)
        if 'rng' in state_input:
            keys = jax.random.split(state_input.rng, n)
            return jax.vmap(lambda k: d.build_state(
                p, seed.replace(rng=k), input=carry))(keys)
        return _tile_state(d.build_state(p, state_input, input=carry), n)

    return init_fn


def _mapped_param_fn(inner: NodeDef, n: int | None) -> Callable:
    """parameterize for transforms that stack a def along a leading axis
    (ensemble members, stack layers): the inner's constructor
    vmapped over n splits of the boundary rng — one independent draw per
    member/layer; the bound input shape resolves the inner's def. A
    non-parametric inner has nothing to stack: the trivial constructor."""
    if not inner.parametric:
        return _trivial_param_fn
    if n is None:
        return inner._param_impl

    def param_fn(ndef, param_input=Struct()):
        carry = _input_or_none(ndef)
        d = inner if carry is None else _resolve(inner, carry)
        if 'rng' not in param_input:
            return d.build_param(param_input)
        keys = jax.random.split(param_input.rng, n)
        bundle = param_input.without('rng')
        return jax.vmap(lambda k: d.build_param(bundle.replace(rng=k)))(keys)

    return param_fn
