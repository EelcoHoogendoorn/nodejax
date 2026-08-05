"""Layer 2: specs — declare the input, derive the rest.

A resolved apply input spec is a pytree with jax.ShapeDtypeStruct leaves:
declared (apply_input_spec=), bound later (with_input), or resolved by a
wiring. The OUT specs (param/state/output) derive by jax.eval_shape over the
contract fns — so they can never go stale, and shape-dependent code
(matmul, reductions) traces exactly.

materialize() turns a spec into zeros, so 'state shaped by the input' (EMA
trees, previous-error registers, feedback seeds) is just an init that reads
its input channel — and a REAL value in that channel primes state from data
through the same mechanism. Factories thread the carry member by member, so
a shape is supplied once at the boundary and flows to every member.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import PyTree
from nodejax.core import Node


def spec(shape: int | tuple = (), dtype: Any = jnp.float32) -> jax.ShapeDtypeStruct:
    """Declare a single array spec; an int is shorthand for a 1-d shape."""
    if isinstance(shape, int):
        shape = (shape,)
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def spec_of(tree: PyTree) -> PyTree:
    """The spec of a pytree: ShapeDtypeStruct leaves for concrete values;
    existing spec leaves pass through."""
    return jax.tree.map(
        lambda leaf: leaf if isinstance(leaf, jax.ShapeDtypeStruct)
        else jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.result_type(leaf)), tree)


def materialize(tree: PyTree) -> PyTree:
    """Turn a spec (or value) into a materialized value: spec leaves become zeros,
    concrete leaves pass through unchanged — so a real value primes
    state with data where a spec only supplies shape."""
    return jax.tree.map(
        lambda leaf: jnp.zeros(leaf.shape, leaf.dtype)
        if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, tree)


def initialize(node: Node, *, input: PyTree | None = None, rng: Any | None = None) -> PyTree:
    """Construct a node's initial state, binding an input spec (given, or
    the def's already-bound spec) and offering an rng key. The spec
    reaches every pipe member via composite init."""
    if input is not None:
        node = node.with_input(input)
    return node.init(Struct(rng=rng) if rng is not None else Struct())


def meta(node: Node, input: PyTree | None = None, rng: Any | None = None) -> Struct:
    """Derive a node's full Meta from its bound param and an input spec
    (given, or declared on the def). All derivation is jax.eval_shape —
    abstract, exact, and computed from the functions that will actually
    run, so it cannot go stale.

    Returns the complete metadata surface: Struct(name, parametric, cyclic,
    and the six specs — param_input_spec / state_input_spec / apply_input_spec
    (IN: what a caller supplies, rigid from the signatures) and param_spec /
    state_spec / output_spec (OUT: what the node produces, by eval_shape).
    The OUT specs (and apply_input_spec) are None where no input spec is
    available to derive them from."""
    nd = node.ndef
    if input is None:
        input = nd.apply_input_spec
    ispec = spec_of(input) if input is not None else None
    seed = Struct(rng=rng) if rng is not None else Struct()

    if ispec is None:
        state = jax.eval_shape(lambda p: nd.build_state(p, seed), node.param)
        output = None
    else:
        nd = nd.with_input(ispec)
        cyc = nd.cyclic
        state = jax.eval_shape(
            lambda p, i: nd.build_state(p, seed, input=i if cyc else None), node.param, ispec)

        def full(p, i):
            s = nd.build_state(p, seed, input=i if cyc else None)
            _, out = nd.apply_fn(p, s, i)
            return out

        output = jax.eval_shape(full, node.param, ispec)

    return Struct(name=nd.name, parametric=nd.parametric, cyclic=nd.cyclic,
                  param_input_spec=nd.param_input_spec,      # IN: what a caller supplies
                  state_input_spec=nd.state_input_spec,      # (rigid, from signatures)
                  apply_input_spec=ispec,
                  param_spec=spec_of(node.param),            # OUT: what the node produces
                  state_spec=state, output_spec=output)      # (derived by eval_shape)
