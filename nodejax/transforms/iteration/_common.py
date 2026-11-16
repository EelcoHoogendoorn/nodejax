"""Sequential `jax.lax.scan` execution and priming helpers shared across iteration transforms.

Unlike parallel `vmap` transforms where every copy sees the same outer input,
sequential composition (`stack`) feeds position `k` from position `k - 1`. Both
`scanned_apply` and `scanned_initialize` thread the running activation across
`length` positions while splitting PRNG keys per position and stacking per-layer
states and `Aux` outputs along leading axis 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.contract import Contract
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.types import PyTree
from nodejax.struct import Struct


def scanned_apply(inner: Contract,
                  param, state, input, rng: MaybeKeyStream, *,
                  scanned_params: bool,
                  length: int) -> tuple[PyTree, PyTree]:
    """Chains `inner.apply` across `length` sequential positions via `jax.lax.scan`.

    When `scanned_params` is `True`, `param` supplies one slice per position;
    otherwise one parameter tree is shared across all `length` steps. Per-step
    states and `Aux` payloads stack along leading axis 0 while clean outputs
    pass forward as the next position's input.
    """
    from nodejax.core.binding import split_aux

    n = length
    rngs, _ = rng.axis(inner.apply_takes_rng, n)
    first = inner.intake(input)

    # An empty MaybeKeyStream has no array leaves and broadcasts across the scan.
    if scanned_params:
        scanned, unpack = ((param, state, rngs),
                           lambda x: (x[0], x[1], x[2]))
    else:
        scanned, unpack = ((state, rngs),
                           lambda x: (param, x[0], x[1]))

    def step(carry, xs):
        p, s, child_rng = unpack(xs)
        new_state, out = inner.apply(
            p, s, inner.feed(carry), child_rng)
        clean_out, aux = split_aux(out)
        return clean_out, (new_state, aux)

    out, (new_states, auxs) = jax.lax.scan(step, first, scanned, length=n)
    return new_states, (out, auxs) if auxs is not None else out


def scanned_initialize(inner: Contract,
                       param, state_input: Struct, input, rng: MaybeKeyStream, *,
                       count: int, scanned_params: bool) -> PyTree:
    """Primes `count` sequential layer states by propagating `input` through preceding layers.

    Each layer primes its state from the activation produced by the previous
    layer's `apply` call rather than from the raw outer `input`, matching the
    shapes and values seen during forward execution.
    """
    contract = inner.for_input(input)

    init_rngs, _ = rng.axis(contract.init_takes_rng, count)
    apply_rngs, _ = rng.axis(
        count > 1 and contract.apply_takes_rng, count - 1)
    value = input

    def at(stream, index):
        return jax.tree.map(lambda leaf: leaf[index], stream)

    def layer_param(index):
        return (jax.tree.map(lambda leaf: leaf[index], param)
                if scanned_params else param)

    def initialize_layer(index, layer_input):
        row = layer_param(index)
        return row, contract.prime(
            row, state_input, layer_input, at(init_rngs, index))

    if count == 1:
        _, final_state = initialize_layer(0, value)
        return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), final_state)

    from nodejax.core.binding import split_aux

    def initialize_and_step(layer_input, index):
        row, state = initialize_layer(index, layer_input)
        _, stepped = contract.apply(
            row, state, inner.feed(layer_input),
            at(apply_rngs, index))
        output, aux = split_aux(stepped)
        return output, state

    value, prefix_states = jax.lax.scan(
        initialize_and_step,
        value,
        jnp.arange(count - 1),
    )
    _, final_state = initialize_layer(count - 1, value)
    return jax.tree.map(
        lambda prefix, final: jnp.concatenate(
            (prefix, jnp.expand_dims(final, 0)), axis=0),
        prefix_states,
        final_state,
    )
