"""Defines signal-conditioning nonlinearities, discrete filters, and one-step delay registers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.core.types import PyTree


@node
def Quantize(resolution: float) -> PNode:
    """Rounds each signal leaf to the nearest multiple of `resolution`."""
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
    """Suppresses signal magnitudes up to `param.threshold` and shifts larger magnitudes continuously toward zero."""
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
def RateLimit(dt: float) -> Node:
    """Clamps per-step signal changes to `[-param.max_rate * dt, param.max_rate * dt]` relative to the previous output."""
    def param(max_rate: PyTree) -> Struct:
        return Struct(max_rate=max_rate)

    def init(node, param) -> PyTree:
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(param, state, input) -> tuple[PyTree, PyTree]:
        limit = param.max_rate * dt
        new = jax.tree.map(lambda last, x: last + jnp.clip(x - last, -limit, limit),
                           state, input)
        return new, new

    return Leaf(apply, init=init, param=param)


@node
def Clamp() -> Node:
    """Clips each signal leaf elementwise to `[-param.limit, param.limit]`."""
    def param(limit: PyTree) -> Struct:
        return Struct(limit=limit)

    def apply(param, input) -> PyTree:
        return jax.tree.map(lambda x: jnp.clip(x, -param.limit, param.limit), input)

    return Leaf(apply, param=param)


@node
def ClampNorm() -> Node:
    """Clamps the norm of a structured signal via `input.clamp_norm(param.limit)` while preserving direction."""
    def param(limit: PyTree) -> Struct:
        return Struct(limit=limit)

    def apply(param, input) -> PyTree:
        return input.clamp_norm(param.limit)

    return Leaf(apply, param=param)


@node
def Delay(warm: bool = False) -> PNode:
    """Emits the previous step's stored signal and stores the current `input` for the next step.

    Cold initialization (`warm=False`) zeros `node.input` so only an input spec is required. Warm initialization (`warm=True`) copies a concrete initial `input` into state. When a `Delay` is read before being fed inside a feedback Composite, forward propagation does not reach it first, so attach an explicit spec via `Delay().with_input(spec)`.
    """
    def init(input) -> PyTree:
        return input

    def init_cold(node) -> PyTree:
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input) -> tuple[PyTree, PyTree]:
        return input, state

    return Leaf(apply, init=init if warm else init_cold)


@node
def Diff(dt: float) -> PNode:
    """Approximates the backward-difference time derivative `(input - previous) / dt`."""
    def init(node) -> PyTree:
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input) -> tuple[PyTree, PyTree]:
        return input, (input - state) / dt

    return Leaf(apply, init=init)


@node
def EMA(dt: float, warm: bool = False) -> Node:
    """Filters any PyTree signal through a discrete first-order lag parameterized by time constant `tau` in seconds.

    Computes step weight `alpha = 1 / (param.tau / dt + 1)`. Cold initialization (`warm=False`) zeros `node.input`; warm initialization (`warm=True`) requires a concrete initial `input` and starts state at that value.
    """
    def param(tau: PyTree) -> Struct:
        return Struct(tau=tau)

    def init(input) -> PyTree:
        return input

    def init_cold(node) -> PyTree:
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(param, state, input) -> tuple[PyTree, PyTree]:
        alpha = 1.0 / (param.tau / dt + 1.0)
        new = jax.tree.map(lambda s, x: s * (1.0 - alpha) + x * alpha, state, input)
        return new, new

    return Leaf(apply, init=init if warm else init_cold, param=param)


@node
def MovingAverage(window: int, warm: bool = False) -> PNode:
    """Averages the most recent `window` samples of any PyTree signal.

    Cold initialization (`warm=False`) fills the `window`-step history buffer with zeros matching `node.input`. Warm initialization (`warm=True`) requires a concrete initial `input` and broadcasts it across all `window` slots.
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
def Blend(dt: float) -> Node:
    """Interpolates `fast * alpha + slow * (1 - alpha)` with weight `alpha = 1 / (param.tau / dt + 1)`."""
    def param(tau: PyTree) -> Struct:
        return Struct(tau=tau)

    def apply(param, fast: PyTree, slow: PyTree) -> PyTree:
        alpha = 1.0 / (param.tau / dt + 1.0)
        return fast * alpha + slow * (1.0 - alpha)

    return Leaf(apply, param=param)


@node
def Gain() -> Node:
    """Multiplies `input` by `param.scale`."""
    def param(scale: PyTree) -> Struct:
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input) -> PyTree:
        return param.scale * input

    return Leaf(apply, param=param)


@node
def Integrator() -> Node:
    """Accumulates `state * (1 - param.decay) + input` at each step.

    Defaults to `decay=0.0` for a pure accumulator; nonzero `decay` implements a leaky integrator (`decay = dt / tau` discretizes an Ornstein-Uhlenbeck drift when driven by white noise). Compose `Gain() >> Integrator()` when input scaling is needed.
    """
    def param(decay: PyTree = 0.0) -> Struct:
        return Struct(decay=jnp.asarray(decay))

    def init(param) -> float:
        return 0.0

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        new = state * (1.0 - param.decay) + input
        return new, new

    return Leaf(apply, param=param, init=init)


@node
def Walker() -> Node:
    """Advances a scalar random-walk position by `input` plus Gaussian noise of standard deviation `param.sigma`."""
    def param(sigma: PyTree) -> Struct:
        return Struct(sigma=jnp.asarray(sigma))

    def init(param, rng) -> Struct:
        return Struct(x=jnp.asarray(0.0), rng=rng)

    def apply(param, state, input) -> tuple[Struct, jax.Array]:
        new_x = state.x + input + param.sigma * jax.random.normal(state.rng)
        return state.replace(x=new_x), new_x

    return Leaf(apply, param=param, init=init)
