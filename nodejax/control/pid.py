"""PID and PD controller primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def PD(dt):
    """PD controller: parametric cyclic node. Its state (the previous error,
    for the D term) is shaped by the init input value — no shape static."""
    def param(kp, kd):
        return Struct(kp=jnp.asarray(kp), kd=jnp.asarray(kd))

    def init(node, param):
        return jnp.zeros_like(node.input)

    def apply(param, state, input):
        force = param.kp * input + param.kd * (input - state) / dt
        return input, force

    return Leaf(apply, param=param, init=init)


@node
def PID(dt, dwrap=None):
    """The PID core: error -> command. Numerics:
    back-calculation anti-windup (I-term clamped in output units),
    exp(-decay*dt) integrator decay, optional wrap on the derivative's
    error difference (dwrap) for rotary signals. Signal-polymorphic:
    state shapes derive from the init input value (scalar or DQ)."""
    def param(kp, ki, kd=0.0, decay=0.0, integral_limit=1e9):
        return Struct(kp=kp, ki=ki, kd=kd,
                      decay=decay, integral_limit=integral_limit)

    def init(node, param):
        zeros = jax.tree.map(jnp.zeros_like, node.input)
        return Struct(integral=zeros, last_error=zeros)

    def apply(param, state, input):
        error = input
        p_term = error * param.kp

        integral_raw = (state.integral + error * dt) * jnp.exp(-param.decay * dt)
        i_term = jax.tree.map(lambda x: jnp.clip(x, -param.integral_limit, param.integral_limit),
                              integral_raw * param.ki)
        # back-calculation anti-windup: store the integral the clamped
        # I-term implies (ki=0 keeps the raw integral)
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
