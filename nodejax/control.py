"""Control-loop combinators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core import NodeDef
from nodejax.authoring import node_def, derive


def closed_loop(pipe: NodeDef) -> NodeDef:
    """Wrap an actuation pipe in unit feedback: reference in ->
    measurement out, with the pipe mapping tracking error -> actuation
    -> measurement. State is the pipe's own plus the fed-back
    measurement register."""
    def init(ndef, param):
        return Struct(inner=pipe.build_state(param, input=ndef.input),
                      last=jnp.zeros_like(ndef.input))

    def apply(param, state, input):
        new_inner, out = pipe.apply_fn(param, state.inner, input - state.last)
        return Struct(inner=new_inner, last=out), out

    return derive(pipe, apply=apply, init=init, name='loop')


def observed_loop(pipe: NodeDef, belief0) -> NodeDef:
    """closed_loop with an in-loop observer riding the plant side —
    the indirect-adaptive-control block. The pipe maps
    Struct(error, belief) -> Struct(output=<measurement>, belief=<the
    observer's current estimate>): the measurement feeds back
    subtractively as tracking error, while the belief feeds FORWARD to
    the pipe's head unmodified — knowledge is offered, never
    subtracted from a reference.

    The loop is coupled to its observer by this output contract: the
    pipe's TAIL must be an observer-wrapped plant emitting
    Struct(output, belief) — identified() in the examples is the
    producer — and belief0 must match the observer's belief
    in shape and meaning: its SHAPE seeds the belief register (zeros
    until the first real estimate lands) and the init-time spec
    propagation. Each step's belief is read where the plant is, and
    consumed by the controller one step later."""
    def init(ndef, param):
        return Struct(inner=pipe.build_state(param, input=Struct(error=ndef.input, belief=belief0)),
                      last=jnp.zeros_like(ndef.input),
                      belief=jax.tree.map(jnp.zeros_like, belief0))

    def apply(param, state, input):
        new_inner, out = pipe.apply_fn(
            param, state.inner, Struct(error=input - state.last, belief=state.belief))
        return Struct(inner=new_inner, last=out.output, belief=out.belief), out.output

    return derive(pipe, apply=apply, init=init, name='oloop')
