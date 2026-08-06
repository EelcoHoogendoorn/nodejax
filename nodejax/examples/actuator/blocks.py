"""Controller building blocks.

THE PID IS A PIPELINE: optional behavior is a pipe member you include
or don't —

    position_ctrl = wrap() >> pid(dt, dwrap=angle) >> clamp()
    current_ctrl  = pid(dt)                       # bare core, DQ signals

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


@ambient
def pid_def(dt, dwrap=None):
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


@ambient
def rate_limit_def(dt):
    """Optional pipeline stage: limit the output slew rate, in units
    per SECOND."""
    def param(max_rate):
        return Struct(max_rate=max_rate)

    def init(ndef, param):
        return jax.tree.map(jnp.zeros_like, ndef.input)     # delayed signal; previous output

    def apply(self, state, input):
        limit = self.max_rate * dt
        new = jax.tree.map(lambda last, x: last + jnp.clip(x - last, -limit, limit),
                    state, input)
        return new, new

    return node_def(apply, init=init, param=param, name='rate_limit')


def clamp_def():
    """Optional pipeline stage: symmetric elementwise output limit."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return jax.tree.map(lambda x: jnp.clip(x, -self.limit, self.limit), input)

    return node_def(apply, param=param, name='clamp')


def clamp_norm_def():
    """Optional pipeline stage: limit a vector's MAGNITUDE (norm),
    preserving direction — the current/voltage limiter, versus
    clamp_def's per-component box. Wraps the input's clamp_norm."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return input.clamp_norm(self.limit)

    return node_def(apply, param=param, name='clamp_norm')


def wrap_def():
    """Optional pipeline stage: angle-wrap the incoming error (rotary)."""
    return node_def(lambda input: angle_wrap(input), name='wrap')


@ambient
def observer_def(dt):
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


def encoder_def(resolution=2000, noise_std=0.1, phase_offset=0.0):
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


def current_sensor_def(noise_std=0.1):
    """DQ current sensor with gaussian noise; rng as state."""
    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        nd, nq = jax.random.normal(state.rng, shape=(2,)) * noise_std
        return state, input + DQ(nd, nq)

    return node_def(apply, init=init, name='current_sensor')


@ambient
def ema_def(dt):
    """First-order low-pass, tau in seconds; signal-polymorphic (state
    from the init input value)."""
    def param(tau):
        return Struct(tau=tau)

    def init(ndef, param):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(self, state, input):
        alpha = 1.0 / (self.tau / dt + 1.0)
        new = jax.tree.map(lambda s, x: s * (1.0 - alpha) + x * alpha, state, input)
        return new, new

    return node_def(apply, init=init, param=param, name='ema')


def blend_def(dt):
    """Blend two signals by a one-pole weight: fast * alpha + slow *
    (1 - alpha), alpha = 1/(tau/dt + 1); tau=0 is pure fast. A leaf
    node — the two signals its input, the weight (tau) its param."""
    def param(tau):
        return Struct(tau=tau)

    def apply(self, fast, slow):
        alpha = 1.0 / (self.tau / dt + 1.0)
        return fast * alpha + slow * (1.0 - alpha)

    return node_def(apply, param=param, name='blend')


def noisy_def(noise_std=0.1):
    """Additive gaussian measurement noise on a scalar; rng as state.
    Compose sensors as pipelines: voltage_est = noisy() >> ema()."""
    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        return state, input + jax.random.normal(state.rng) * noise_std

    return node_def(apply, init=init, name='noisy')


def bag_def(**fields):
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
            raise TypeError(f'bag has no field(s) {sorted(unknown)}; declared: {sorted(fields)}')
        return Struct(**{**fields, **given})

    def apply(param, input):
        return param

    return node_def(apply, param=param, name='bag')


def delay_def(value):
    """One-tick memory member: emits the value stored last step, stores
    its input. value declares the stored value's default shape; a
    composite's init discovery reshapes it from the wiring."""
    def init(ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(state, input):
        return input, state

    return node_def(apply, init=init, apply_input_spec=value, name='delay')


@ambient
def diff_def(dt):
    """Discrete time derivative: (input - previous) / dt, the previous
    value the node's state. Shape-generic: the node is always fed, so
    the wiring resolves its spec from the call site (a delay declares a
    default value instead because it is read before it is fed)."""
    def init(ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(state, input):
        return input, (input - state) / dt

    return node_def(apply, init=init, name='diff')
