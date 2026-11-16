"""Defines rotary angle wrapping, state observers, noisy sensors, and parameter-bag Leaves for FOC control."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, PNode, PyTree, node, Leaf

from examples.actuator.dq import DQ
from examples.actuator.utils import wrap as angle_wrap


@node
def wrap() -> PNode:
    """Wraps rotary angle errors into `[-pi, pi]` as an inline pipeline stage."""
    return Leaf(lambda input: angle_wrap(input))


@node
def Observer(dt: float) -> Node:
    """Tracks rotor position and velocity with an angle-wrapped predict-correct filter.

    Warm-initializes `Struct(position=input, velocity=0.0, measured=input)` at the first measured angle. Differences consecutive raw measurements (`angle_wrap(measured - state.measured) / dt`) rather than filtered positions so position-filter lag does not inject acceleration spikes into `velocity`.
    """
    def param(tau_pos: float = 0.1, tau_vel: float = 0.1) -> Struct:
        return Struct(tau_pos=tau_pos, tau_vel=tau_vel)

    def init(param, input) -> Struct:
        """Initializes position and previous measurement at `input` with zero velocity."""
        return Struct(position=input, velocity=0.0, measured=input)

    def apply(param, state, input) -> tuple[Struct, Struct]:
        measured = input
        a_pos = 1.0 / (1.0 + param.tau_pos)
        a_vel = 1.0 / (1.0 + param.tau_vel)

        predicted = state.position + state.velocity * dt
        position = predicted + a_pos * angle_wrap(measured - predicted)
        velocity_raw = angle_wrap(measured - state.measured) / dt
        velocity = state.velocity * (1.0 - a_vel) + a_vel * velocity_raw

        return (Struct(position=position, velocity=velocity, measured=measured),
                Struct(position=position, velocity=velocity))

    return Leaf(apply, init=init, param=param)


@node
def ClampNorm() -> Node:
    """Clamps a `DQ` vector to Euclidean radius `limit` via `DQ.clamp_norm`."""
    def param(limit: float = 1.0) -> Struct:
        return Struct(limit=limit)

    def apply(param, input: DQ) -> DQ:
        return input.clamp_norm(param.limit)

    return Leaf(apply, param=param)


@node
def Encoder() -> Node:
    """Simulates a quantized absolute rotary encoder with phase offset and additive count noise."""
    def param(resolution: float = 2000.0, noise_std: float = 0.1,
              phase_offset: float = 0.0) -> Struct:
        return Struct(resolution=resolution, noise_std=noise_std,
                      phase_offset=phase_offset)

    def apply(param, input, rng) -> jax.Array:
        rad_to_counts = param.resolution / (2.0 * jnp.pi)
        counts = (input + param.phase_offset) * rad_to_counts
        counts = counts + jax.random.normal(rng.next()) * param.noise_std
        return jnp.round(counts) / rad_to_counts - param.phase_offset

    return Leaf(apply, param=param)


@node
def CurrentSensor() -> Node:
    """Adds independent Gaussian measurement noise of standard deviation `noise_std` to a `DQ` current pair."""
    def param(noise_std: float = 0.1) -> Struct:
        return Struct(noise_std=noise_std)

    def apply(param, input: DQ, rng) -> DQ:
        noise_d, noise_q = (
            jax.random.normal(rng.next(), shape=(2,)) * param.noise_std)
        return input + DQ(noise_d, noise_q)

    return Leaf(apply, param=param)


@node
def Noisy() -> Node:
    """Adds scalar Gaussian measurement noise of standard deviation `noise_std` to `input`."""
    def param(noise_std: float = 0.1) -> Struct:
        return Struct(noise_std=noise_std)

    def apply(param, input, rng) -> jax.Array:
        return input + jax.random.normal(rng.next()) * param.noise_std

    return Leaf(apply, param=param)


@node
def Bag(**fields: PyTree) -> Node:
    """Stores a `Struct` of named scalar or array parameters for an enclosing Composite to read across multiple expressions.

    Accepts default field values at factory construction and overrides declared fields during `parameterize` while rejecting unknown field names.
    """
    def param(**given: PyTree) -> Struct:
        unknown = set(given) - set(fields)
        if unknown:
            raise TypeError(f'Bag has no field(s) {sorted(unknown)}; declared: {sorted(fields)}')
        return Struct(**{**fields, **given})

    def apply(param, input) -> Struct:
        return param

    return Leaf(apply, param=param)
