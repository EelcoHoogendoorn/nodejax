"""Control signal manipulation and filter blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import node
from nodejax.authoring import Leaf
from nodejax.node import Node
from nodejax.types import PyTree


@node
def Quantize(resolution: float) -> Node:
    """Round each signal value to the nearest multiple of `resolution`.

    Resolution fixes the output grid, so it is part of the definition rather
    than a differentiable param. PyTree inputs are quantized leaf by leaf.
    """
    if resolution <= 0.0:
        raise ValueError('resolution must be positive')

    def apply(input) -> PyTree:
        return jax.tree.map(
            lambda value: jnp.round(value / resolution) * resolution,
            input,
        )

    return Leaf(apply)


@node
def Deadband() -> Node:
    """Symmetric deadband with a continuous transition at the threshold.

    Magnitudes up to `threshold` map to zero. Values outside the band have
    the threshold subtracted from their magnitude, preserving sign. The
    threshold is a param and may be scalar or broadcast over each input leaf.
    """
    def param(threshold: PyTree) -> Struct:
        return Struct(threshold=jnp.asarray(threshold))

    def apply(param, input) -> PyTree:
        return jax.tree.map(
            lambda value: jnp.sign(value)
            * jnp.maximum(jnp.abs(value) - param.threshold, 0.0),
            input,
        )

    return Leaf(apply, param=param)


@node(name='rate_limit')
def RateLimit(dt):
    """Optional pipeline stage: limit the output slew rate, in units
    per SECOND."""
    def param(max_rate):
        return Struct(max_rate=max_rate)

    def init(node, param):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(self, state, input):
        limit = self.max_rate * dt
        new = jax.tree.map(lambda last, x: last + jnp.clip(x - last, -limit, limit),
                           state, input)
        return new, new

    return Leaf(apply, init=init, param=param)


@node
def Clamp():
    """Optional pipeline stage: symmetric elementwise output limit."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return jax.tree.map(lambda x: jnp.clip(x, -self.limit, self.limit), input)

    return Leaf(apply, param=param)


@node
def ClampNorm():
    """Optional pipeline stage: limit a vector's MAGNITUDE (norm),
    preserving direction."""
    def param(limit):
        return Struct(limit=limit)

    def apply(self, input):
        return input.clamp_norm(self.limit)

    return Leaf(apply, param=param)


@node
def Delay(warm: bool = False):
    """One-tick memory: emits what was stored last step, stores its input.

    Where it BEGINS is the same question EMA answers, and the same static
    answers it: warm primes the register from the first real input, so the
    node requires a value at init; cold begins at zeros of the signal's
    shape. Cold is the default, and it is the whole of the value story. A
    register that must start somewhere other than rest is init-then-set,
    not a constructor argument: init discovers structure and zeros it, and
    a start that takes real work (a settling run, a neutral pose) is the
    caller's, kept out of the rollout being scored.

    A delay placed where the wiring READS IT BEFORE FEEDING IT — a feedback
    register rather than a filter — is never reached by forward shape
    propagation, and a shape it has not got cannot be zeroed. That position
    declares what flows through it the ordinary way, Delay().with_input(spec),
    like any other node whose wiring does not supply its shape."""
    def init(input):
        return input

    def init_cold(node):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init if warm else init_cold)


@node
def Diff(dt):
    """Discrete time derivative: (input - previous) / dt."""
    def init(node):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input):
        return input, (input - state) / dt

    return Leaf(apply, init=init)


@node
def EMA(dt, warm: bool = False):
    """First-order low-pass, tau in seconds; signal-polymorphic.

    `warm` starts the filter AT the first real input rather than at zeros
    — a sensor that samples before it is switched on — and the node then
    requires a real value at init. Cold needs only a shape."""
    def param(tau):
        return Struct(tau=tau)

    def init(input):
        return input

    def init_cold(node):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(self, state, input):
        alpha = 1.0 / (self.tau / dt + 1.0)
        new = jax.tree.map(lambda s, x: s * (1.0 - alpha) + x * alpha, state, input)
        return new, new

    return Leaf(apply, init=init if warm else init_cold, param=param)


@node
def MovingAverage(window: int, warm: bool = False) -> Node:
    """Mean over a fixed number of recent samples.

    Cold initialization fills the history with zeros. With `warm=True`, init
    requires an input value and fills the history with that value. PyTree
    inputs keep one history array per leaf.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError('window must be a positive integer')

    def fill(value: PyTree) -> PyTree:
        value = jnp.asarray(value)
        return jnp.broadcast_to(value, (window,) + value.shape)

    def init(input) -> PyTree:
        return jax.tree.map(fill, input)

    def init_cold(node) -> PyTree:
        return jax.tree.map(lambda value: fill(jnp.zeros_like(value)),
                            node.input)

    def apply(state, input) -> tuple[PyTree, PyTree]:
        state = jax.tree.map(
            lambda history, value: jnp.concatenate(
                (history[1:], jnp.expand_dims(value, axis=0)), axis=0),
            state,
            input,
        )
        output = jax.tree.map(lambda history: jnp.mean(history, axis=0), state)
        return state, output

    return Leaf(apply, init=init if warm else init_cold)


@node
def Blend(dt):
    """Blend two signals by a one-pole weight: fast * alpha + slow * (1 - alpha)."""
    def param(tau):
        return Struct(tau=tau)

    def apply(self, fast, slow):
        alpha = 1.0 / (self.tau / dt + 1.0)
        return fast * alpha + slow * (1.0 - alpha)

    return Leaf(apply, param=param)


@node
def Gain():
    """Proportional block: output = scale * input."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return Leaf(apply, param=param)


@node
def Integrator():
    """Integral block: state accumulates its input, forgetting a fraction
    `decay` of what it holds each step.

    Accumulating is all it does. A scaled integral is Gain() >> Integrator(),
    which is why there is no gain here. The decay cannot be factored out the
    same way, because it multiplies the STATE rather than the input, and that
    is exactly what makes it the integrator's own business.

    decay=0 is the pure accumulator and the default. Nonzero makes it a leaky
    integrator, the block behind every process that wanders but stays bounded:
    fed white noise it IS an Ornstein-Uhlenbeck bias, with decay = dt/tau."""
    def param(decay=0.0):
        return Struct(decay=jnp.asarray(decay))

    def init(param):
        return 0.0

    def apply(param, state, input):
        new = state * (1.0 - param.decay) + input
        return new, new

    return Leaf(apply, param=param, init=init)


@node
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

    return Leaf(apply, param=param, init=init)
