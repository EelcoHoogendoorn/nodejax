"""Control signal manipulation and filter blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def


@ambient
def RateLimit(dt):
    """Optional pipeline stage: limit the output slew rate, in units
    per SECOND."""
    def param(max_rate):
        return Struct(max_rate=max_rate)

    def init(ndef, param):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(self, state, input):
        limit = self.max_rate * dt
        new = jax.tree.map(lambda last, x: last + jnp.clip(x - last, -limit, limit),
                           state, input)
        return new, new

    return node_def(apply, init=init, param=param, name='rate_limit')


def Clamp():
    """Optional pipeline stage: symmetric elementwise output limit."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return jax.tree.map(lambda x: jnp.clip(x, -self.limit, self.limit), input)

    return node_def(apply, param=param, name='clamp')


def ClampNorm():
    """Optional pipeline stage: limit a vector's MAGNITUDE (norm),
    preserving direction."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return input.clamp_norm(self.limit)

    return node_def(apply, param=param, name='clamp_norm')


def Delay(value=0.0):
    """One-tick memory member: emits the value stored last step, stores
    its input."""
    def init(ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(state, input):
        return input, state

    return node_def(apply, init=init, apply_input_spec=value, name='delay')


@ambient
def Diff(dt):
    """Discrete time derivative: (input - previous) / dt."""
    def init(ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(state, input):
        return input, (input - state) / dt

    return node_def(apply, init=init, name='diff')


@ambient
def EMA(dt):
    """First-order low-pass, tau in seconds; signal-polymorphic."""
    def param(tau):
        return Struct(tau=tau)

    def init(input):
        return input

    def apply(self, state, input):
        alpha = 1.0 / (self.tau / dt + 1.0)
        new = jax.tree.map(lambda s, x: s * (1.0 - alpha) + x * alpha, state, input)
        return new, new

    return node_def(apply, init=init, param=param, name='ema')


def Blend(dt):
    """Blend two signals by a one-pole weight: fast * alpha + slow * (1 - alpha)."""
    def param(tau):
        return Struct(tau=tau)

    def apply(self, fast, slow):
        alpha = 1.0 / (self.tau / dt + 1.0)
        return fast * alpha + slow * (1.0 - alpha)

    return node_def(apply, param=param, name='blend')


def Gain():
    """Proportional block: output = scale * input."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return node_def(apply, param=param, name='gain')


def Integrator():
    """Integral block: state accumulates gain * input."""
    def param(gain):
        return Struct(gain=jnp.asarray(gain))

    def init(param):
        return 0.0

    def apply(param, state, input):
        new = state + param.gain * input
        return new, new

    return node_def(apply, param=param, init=init, name='integrator')


def Walker():
    """Random-walk source: stochastic cyclic state via the reserved rng
    field. The apply reads state.rng and never splits or threads it."""
    def param(sigma):
        return Struct(sigma=jnp.asarray(sigma))

    def init(param, rng):
        return Struct(x=jnp.asarray(0.0), rng=rng)

    def apply(param, state, input):
        new_x = state.x + input + param.sigma * jax.random.normal(state.rng)
        return state.replace(x=new_x), new_x

    return node_def(apply, param=param, init=init, name='walker')
