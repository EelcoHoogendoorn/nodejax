"""Shared execution helpers for iteration transforms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.contract import Contract
from nodejax.core.rng import MaybeKeyStream


def scanned_apply(inner: Contract,
                  param, state, input, rng: MaybeKeyStream, *,
                  scanned_params: bool, length: int) -> tuple[Any, Any]:
    """Uniform lax.scan execution over a sequential axis: position k's output
    is position k+1's input, and the per-position states stack.

    `length` is how many positions, and it is REQUIRED because the caller
    has it as a static: stack's depth is a construction argument. It used to default to counting the param rows, which read the
    number off the wrong thing and crashed on a cyclic node with no params,
    which stack accepts by its own guard.

    `scanned_params` says whether params carry that axis, one row per
    position (stack's layers), or one set threads every position (a
    non-parametric inner). The apply-side twin of mapped initialization's
    parameter axis, and the same question: does every member own its
    params. Apply-side rng splits per position, so no two positions share a
    draw. Aux rides the output as it does under vmap: split from the carry
    each step, stacked over the axis, re-emitted as the (output, aux) pair."""
    from nodejax.core.binding import (split_aux)


    n = length
    rngs, _ = rng.axis(inner.apply_takes_rng, n)
    first = inner.intake(input)

    # An empty MaybeKeyStream has no leaves and broadcasts while params or state provide
    # the scan length.
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
                       count: int, scanned_params: bool) -> Any:
    """The inner's init for a SEQUENTIAL axis (stack's layers): one state
    per position, built the way the signal meets it.

    The vmapped sibling serves the parallel transforms, where every member
    legitimately sees the outer input. Positions in sequence do not:
    position k's input is what position k-1 produced, so a supplied value
    threads exactly as serial's init walk threads it, each position's state
    built from the running value and the value advanced by that position's
    own apply. A node that PRIMES therefore boots from the signal as it
    arrives at its depth, not from the raw input n times over.

    `scanned_params` as in scanned_apply: a parametric inner owns a row
    of the param per layer, a non-parametric one has nothing to slice.

    ``inner`` is deliberately supplied at invocation time, as it is to the
    vmapped helpers. A transform's member is replaceable structure; closing
    over the member used to construct the wrapper would make initialization
    operate on a stale definition.
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
        return split_aux(stepped)[0], state

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
