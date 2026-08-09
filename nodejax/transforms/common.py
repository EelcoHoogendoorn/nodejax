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
from nodejax.core import Node, NodeDef, _input_or_none, _resolve, _trivial_param_fn
from nodejax.authoring import KeyStream


_UNBOUND: Any = object()  # sentinel: _split saw a NodeDef, not a bound Node


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


def _mapped_init(inner: NodeDef, n: int | None = None, *,
                 stacked: bool | None = None) -> Callable:
    """init for transforms whose STATE gains a leading member axis: one state
    per member. `stacked` says whether the params carry that axis too —
    ensemble members and stack layers do (each member built from its own
    param row); repeat positions and non-parametric members share one param,
    so the count comes from n. A boundary rng in the seed always splits per
    member — an independent stream each, never a copy; a deterministic seed
    tiles (with any reserved state.rng field split, see _tile_state)."""
    if stacked is None:
        stacked = inner.parametric

    def init_fn(ndef, p, state_input=Struct(), input=None):
        carry = input if input is not None else _input_or_none(ndef)
        d = inner if carry is None else _resolve(inner, carry)
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
