"""PID and PD controller primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def


def PD(dt):
    """PD controller: parametric cyclic node. Its state (the previous error,
    for the D term) is shaped by the init input value — no shape static."""
    def param(kp, kd):
        return Struct(kp=jnp.asarray(kp), kd=jnp.asarray(kd))

    def init(ndef, param):
        return jnp.zeros_like(ndef.input)

    def apply(param, state, input):
        force = param.kp * input + param.kd * (input - state) / dt
        return input, force

    return node_def(apply, param=param, init=init, name='pd')


@ambient
def PID(dt, dwrap=None):
    """The PID core: error -> command. Numerics:
    back-calculation anti-windup (I-term clamped in output units),
    exp(-decay*dt) integrator decay, optional wrap on the derivative's
    error difference (dwrap) for rotary signals. Signal-polymorphic:
    state shapes derive from the init input value (scalar or DQ)."""
    def param(kp, ki, kd=0.0, decay=0.0, integral_limit=1e9):
        return Struct(kp=kp, ki=ki, kd=kd,
                      decay=decay, integral_limit=integral_limit)

    def init(ndef, param):
        zeros = jax.tree.map(jnp.zeros_like, ndef.input)
        return Struct(integral=zeros, last_error=zeros)

    def apply(self, state, input):
        error = input
        p_term = error * self.kp

        integral_raw = (state.integral + error * dt) * jnp.exp(-self.decay * dt)
        i_term = jax.tree.map(lambda x: jnp.clip(x, -self.integral_limit, self.integral_limit),
                              integral_raw * self.ki)
        # back-calculation anti-windup: store the integral the clamped
        # I-term implies (ki=0 keeps the raw integral)
        safe_ki = jnp.where(self.ki != 0.0, self.ki, 1.0)
        integral = jax.tree.map(lambda it, ir: jnp.where(self.ki != 0.0, it / safe_ki, ir),
                                 i_term, integral_raw)

        diff = error - state.last_error
        if dwrap is not None:
            diff = jax.tree.map(dwrap, diff)
        d_term = diff * (self.kd / dt)

        output = p_term + i_term + d_term
        return Struct(integral=integral, last_error=error), output

    return node_def(apply, init=init, param=param, name='pid')
