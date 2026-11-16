"""Compositional IMU sensor pipeline in Equinox.

Equinox represents module parameters as PyTrees while leaving state threading, input priming, and PRNG key routing to user code. Building the five-stage IMU requires repeating the stage topology across `IMUState`, `IMU.init`, and `IMU.step`, plus splitting step keys across stochastic stages by hand; inserting one stateful stage mid-pipe requires four coordinated edits.

Run directly with `python -m examples.comparisons.imu.imu_equinox`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from examples.comparisons.imu.imu_common import (
    DENSITY, DRIFT_DENSITY, DT, RES, TAU, trajectory, verify_imu,
)


class Derivative(eqx.Module):
    dt: float = eqx.field(static=True)

    def __call__(self, last: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x, (x - last) / self.dt


class Noise(eqx.Module):
    density: jax.Array

    def __call__(self, x: jax.Array, key: jax.Array) -> jax.Array:
        return x + self.density * jax.random.normal(key, jnp.shape(x))


class Drift(eqx.Module):
    density: jax.Array
    tau: jax.Array
    dt: float = eqx.field(static=True)

    def __call__(self, bias: jax.Array, x: jax.Array,
                 key: jax.Array) -> tuple[jax.Array, jax.Array]:
        step = self.density * jnp.sqrt(self.dt) * jax.random.normal(key, jnp.shape(x))
        new_bias = bias * (1.0 - self.dt / self.tau) + step
        return new_bias, x + new_bias


class Quantizer(eqx.Module):
    resolution: float = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.round(x / self.resolution) * self.resolution


class IMUState(eqx.Module):
    """Holds one explicit state field per stateful stage."""

    d1_last: jax.Array
    d2_last: jax.Array
    drift_bias: jax.Array


class IMU(eqx.Module):
    d1: Derivative
    d2: Derivative
    noise: Noise
    drift: Drift
    quant: Quantizer

    def init(self, x0: jax.Array) -> IMUState:
        _, v0 = self.d1(jnp.asarray(x0), jnp.asarray(x0))
        return IMUState(
            d1_last=jnp.asarray(x0),
            d2_last=v0,
            drift_bias=jnp.asarray(0.0),
        )

    def step(self, state: IMUState, x: jax.Array,
             key: jax.Array) -> tuple[IMUState, jax.Array]:
        k_noise, k_drift = jax.random.split(key)
        d1_last, v = self.d1(state.d1_last, x)
        d2_last, a = self.d2(state.d2_last, v)
        a = self.noise(a, k_noise)
        drift_bias, a = self.drift(state.drift_bias, a, k_drift)
        a = self.quant(a)
        return IMUState(d1_last=d1_last, d2_last=d2_last, drift_bias=drift_bias), a


def make_imu(density: float = DENSITY, tau: float = TAU,
             drift_density: float = DRIFT_DENSITY) -> IMU:
    return IMU(
        d1=Derivative(DT),
        d2=Derivative(DT),
        noise=Noise(density=jnp.asarray(density)),
        drift=Drift(density=jnp.asarray(drift_density), tau=jnp.asarray(tau), dt=DT),
        quant=Quantizer(RES),
    )


def run(imu: IMU, positions: jax.Array, seed: int = 0) -> jax.Array:
    state = imu.init(positions[0])
    keys = jax.random.split(jax.random.PRNGKey(seed), positions.shape[0])
    _, accel = jax.lax.scan(
        lambda s, elem: imu.step(s, elem[0], elem[1]),
        state,
        (positions, keys),
    )
    return accel


def main() -> None:
    _, positions = trajectory()
    pos = jnp.asarray(positions)
    accel = run(make_imu(), pos, seed=0)
    quiet = run(make_imu(density=0.0, drift_density=0.0), pos, seed=0)
    verify_imu('equinox', pos, accel, quiet)


if __name__ == '__main__':
    main()
