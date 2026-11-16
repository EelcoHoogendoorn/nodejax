"""Defines discrete proportional-derivative (`PD`) and proportional-integral-derivative (`PID`) controllers."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import Node
from nodejax.core.types import PyTree


@node
def PD(dt: float) -> Node:
    """Computes `kp * error + kd * (error - previous_error) / dt` with previous-error state shaped from `node.input`."""
    def param(kp: PyTree, kd: PyTree) -> Struct:
        return Struct(kp=jnp.asarray(kp), kd=jnp.asarray(kd))

    def init(node, param) -> PyTree:
        return jnp.zeros_like(node.input)

    def apply(param, state, input) -> tuple[PyTree, PyTree]:
        force = param.kp * input + param.kd * (input - state) / dt
        return input, force

    return Leaf(apply, param=param, init=init)


@node
def PID(dt: float, dwrap: Callable[[PyTree], PyTree] | None = None) -> Node:
    """Maps tracking `error` to actuation command with exponential integral decay and back-calculation anti-windup.

    Shapes `Struct(integral=..., last_error=...)` from `node.input`. Multiplies the accumulated integral by `exp(-param.decay * dt)` each step, clips `integral * param.ki` to `[-param.integral_limit, param.integral_limit]`, stores the implied clamped integral back into state when `ki != 0`, and optionally wraps `error - state.last_error` through `dwrap` on periodic coordinates.
    """
    def param(
            kp: PyTree, ki: PyTree, kd: PyTree = 0.0,
            decay: PyTree = 0.0, integral_limit: PyTree = 1e9,
    ) -> Struct:
        return Struct(kp=kp, ki=ki, kd=kd,
                      decay=decay, integral_limit=integral_limit)

    def init(node, param) -> Struct:
        zeros = jax.tree.map(jnp.zeros_like, node.input)
        return Struct(integral=zeros, last_error=zeros)

    def apply(param, state, input) -> tuple[Struct, PyTree]:
        error = input
        p_term = error * param.kp

        integral_raw = (state.integral + error * dt) * jnp.exp(-param.decay * dt)
        i_term = jax.tree.map(lambda x: jnp.clip(x, -param.integral_limit, param.integral_limit),
                              integral_raw * param.ki)
        # Back-calculation anti-windup stores the integral implied by the clamped I-term (ki=0 keeps integral_raw).
        safe_ki = jnp.where(param.ki != 0.0, param.ki, 1.0)
        integral = jax.tree.map(lambda it, ir: jnp.where(param.ki != 0.0, it / safe_ki, ir),
                                 i_term, integral_raw)

        diff = error - state.last_error
        if dwrap is not None:
            diff = jax.tree.map(dwrap, diff)
        d_term = diff * (param.kd / dt)

        output = p_term + i_term + d_term
        return Struct(integral=integral, last_error=error), output

    return Leaf(apply, init=init, param=param)
