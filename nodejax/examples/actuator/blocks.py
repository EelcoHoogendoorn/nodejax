"""Controller building blocks.

THE PID IS A PIPELINE: optional behavior is a pipe member you include
or don't —

    position_ctrl = wrap() >> PID(dt, dwrap=angle) >> Clamp()
    current_ctrl  = PID(dt)                       # bare core, DQ signals

The core is signal-polymorphic with no subclassing: its integral and
last-error states derive from the init input value, so the same def
runs scalars or DQ pairs.

Numerics: rate limits in units/second, integrator decay exp(-decay*dt),
anti-windup by clamping the I-term in output units and back-converting,
wrap-aware derivative. Sensors are cyclic nodes carrying rng state
(auto-advanced): one key enters at init and the streams run themselves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import ambient, node_def

from nodejax.examples.actuator.dq import DQ
from nodejax.examples.actuator.utils import wrap as angle_wrap


def wrap():
    """Optional pipeline stage: angle-wrap the incoming error (rotary)."""
    return node_def(lambda input: angle_wrap(input), name='wrap')


@ambient
def Observer(dt):
    """Predict-correct EMA observer for position and velocity, with
    angle wrapping; time constants are params (per-step alpha =
    1/(1+tau))."""
    def param(tau_pos=0.1, tau_vel=0.1):
        return Struct(tau_pos=tau_pos, tau_vel=tau_vel)

    def init(param, position=0.0):
        return Struct(position=position, velocity=0.0)

    def apply(self, state, input):
        measured = input
        a_pos = 1.0 / (1.0 + self.tau_pos)
        a_vel = 1.0 / (1.0 + self.tau_vel)

        predicted = state.position + state.velocity * dt
        position = predicted + a_pos * angle_wrap(measured - predicted)
        velocity_raw = angle_wrap(measured - state.position) / dt
        velocity = state.velocity * (1.0 - a_vel) + a_vel * velocity_raw

        new = Struct(position=position, velocity=velocity)
        return new, new

    return node_def(apply, init=init, param=param, name='observer')


def Encoder(resolution=2000, noise_std=0.1, phase_offset=0.0):
    """Absolute encoder: quantization + count noise; rng as state."""
    rad_to_counts = resolution / (2.0 * jnp.pi)

    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        counts = (input + phase_offset) * rad_to_counts
        counts = counts + jax.random.normal(state.rng) * noise_std
        measured = jnp.round(counts) / rad_to_counts - phase_offset
        return state, measured

    return node_def(apply, init=init, name='encoder')


def CurrentSensor(noise_std=0.1):
    """DQ current sensor with gaussian noise; rng as state."""
    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        nd, nq = jax.random.normal(state.rng, shape=(2,)) * noise_std
        return state, input + DQ(nd, nq)

    return node_def(apply, init=init, name='current_sensor')


def Noisy(noise_std=0.1):
    """Additive gaussian measurement noise on a scalar; rng as state.
    Compose sensors as pipelines: voltage_est = Noisy() >> EMA(dt)."""
    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        return state, input + jax.random.normal(state.rng) * noise_std

    return node_def(apply, init=init, name='noisy')


def Bag(**fields):
    """A leaf holding raw params whose behavior is DIFFUSE.

    The one right home for params that are not themselves nodes is a
    leaf — the irreducible bottom of a node tree, where behavior turns
    into actual arrays. Most such params have a clean transform (a
    clamp, a blend, a weighted sum) and are ordinary leaf nodes with a
    real apply. This is the leaf for the remaining case: a coefficient
    an enclosing composite reads across many expressions, with no single
    apply surface of its own to name — so its apply trivially echoes the
    bundle and the behavior lives in the reader. Factory fields are the
    construction defaults; parameterize overrides declared fields
    (unknown fields rejected)."""
    def param(**given):
        unknown = set(given) - set(fields)
        if unknown:
            raise TypeError(f'Bag has no field(s) {sorted(unknown)}; declared: {sorted(fields)}')
        return Struct(**{**fields, **given})

    def apply(param, input):
        return param

    return node_def(apply, param=param, name='bag')
