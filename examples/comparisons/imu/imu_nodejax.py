"""Compositional IMU sensor pipeline in NodeJAX.

Five stages compose into a single `>>` pipeline (`Derivative >> Derivative >> Noise >> Drift >> Quantize`). `Derivative` primes its previous-sample state from the initial input at `initialize`, `Noise` and `Drift` draw from call-time `KeyStream` splits without storing PRNG keys in `state`, and `sensor.scan` threads the derived state container and step keys across the trajectory.

Run directly with `python -m examples.comparisons.imu.imu_nodejax`.
"""

import jax
import jax.numpy as jnp

from nodejax import KeyStream, Leaf, Node, Struct, node
from nodejax.control import Quantize

from examples.comparisons.imu.imu_common import (
    DENSITY, DRIFT_DENSITY, DT, RES, TAU, trajectory, verify_imu,
)


@node
def Derivative(dt: float) -> Node:
    """Discrete time derivative primed from the initial input sample."""
    def init(input) -> jax.Array:
        return jnp.asarray(input)

    def apply(state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        return input, (input - state) / dt

    return Leaf(apply, init=init)


@node
def Noise() -> Node:
    """Additive Gaussian white noise with trainable spectral density."""
    def param(density: float) -> Struct:
        return Struct(density=jnp.asarray(density))

    def apply(param: Struct, input: jax.Array, rng: KeyStream) -> jax.Array:
        return input + param.density * jax.random.normal(rng.next(), jnp.shape(input))

    return Leaf(apply, param=param)


@node
def Drift(dt: float) -> Node:
    """Additive Ornstein-Uhlenbeck bias with carried bias state and call-time RNG."""
    def param(density: float, tau: float) -> Struct:
        return Struct(density=jnp.asarray(density), tau=jnp.asarray(tau))

    def init() -> jax.Array:
        return jnp.asarray(0.0)

    def apply(param: Struct, state: jax.Array, input: jax.Array,
              rng: KeyStream) -> tuple[jax.Array, jax.Array]:
        step = param.density * jnp.sqrt(dt) * jax.random.normal(rng.next(), jnp.shape(input))
        bias = state * (1.0 - dt / param.tau) + step
        return bias, input + bias

    return Leaf(apply, param=param, init=init)


def make_imu(density: float = DENSITY, tau: float = TAU,
             drift_density: float = DRIFT_DENSITY) -> Node:
    """Build the five-stage IMU sensor pipeline."""
    return (Derivative(DT)
            >> Derivative(DT)
            >> Noise()(density=density)
            >> Drift(DT)(density=drift_density, tau=tau)
            >> Quantize(RES))


def run(imu: Node, positions: jax.Array, seed: int = 0) -> jax.Array:
    """Initialize `imu` from `positions[0]` and scan across `positions`."""
    sensor = imu.with_input(positions[0]).parameterize().initialize(input=positions[0])
    _, accel = sensor.scan(positions, rng=jax.random.PRNGKey(seed))
    return accel


def main() -> None:
    _, positions = trajectory()
    pos = jnp.asarray(positions)
    imu = make_imu()
    accel = run(imu, pos, seed=0)
    assert jnp.allclose(accel, run(imu, pos, seed=0))
    assert not jnp.allclose(accel, run(imu, pos, seed=1))
    quiet = run(make_imu(density=0.0, drift_density=0.0), pos, seed=0)
    verify_imu('nodejax', pos, accel, quiet)


if __name__ == '__main__':
    main()
