"""Control loop feedback combinators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core import NodeDef, hoist_rng
from nodejax.authoring import derive


def feedback(pipe: NodeDef, last: float = 0.0) -> NodeDef:
    """Close the loop — a user-land control transform: the wrapped node maps
    tracking error -> output, and feedback supplies error = reference - the
    previous output. Cyclic -> cyclic; param meaning unchanged."""
    def init(ndef, param, state=Struct()):
        seed = state.inner if 'inner' in state else Struct()
        if 'rng' in state:
            seed = seed.replace(rng=state.rng)
        if ndef.apply_input_spec is None:
            return Struct(inner=pipe.build_state(param, seed),
                          last=jax.tree.map(jnp.asarray, last))
        return Struct(inner=pipe.build_state(param, seed, input=ndef.input),
                      last=jax.tree.map(jnp.zeros_like, ndef.input))

    def apply(param, state, input):
        error = jax.tree.map(jnp.subtract, input, state.last)
        new_inner, out = pipe.apply_fn(param, state.inner, error)
        return Struct(inner=new_inner, last=out), out

    seed_spec = hoist_rng(dict(inner=pipe.state_input_spec if pipe.cyclic else Struct()))
    return derive(pipe, apply=apply, init=init, state_input_spec=seed_spec,
                  name=f'feedback({pipe.name})')


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
    the indirect-adaptive-control block."""
    def init(ndef, param):
        return Struct(inner=pipe.build_state(param, input=Struct(error=ndef.input, belief=belief0)),
                      last=jnp.zeros_like(ndef.input),
                      belief=jax.tree.map(jnp.zeros_like, belief0))

    def apply(param, state, input):
        new_inner, out = pipe.apply_fn(
            param, state.inner, Struct(error=input - state.last, belief=state.belief))
        return Struct(inner=new_inner, last=out.output, belief=out.belief), out.output

    return derive(pipe, apply=apply, init=init, name='oloop')
