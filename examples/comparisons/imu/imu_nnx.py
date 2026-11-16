"""Compositional IMU sensor pipeline in Flax NNX.

NNX mutates `nnx.Variable` attributes in place during `__call__` and carries non-`Param` variables through `nnx.scan` via `nnx.StateAxes`. Because `nnx.Sequential` composes calls without composing initialization, `IMU.__init__` wires the initial sample `x0` into `d1`, `0.0` into `d2`, and named `nnx.Rngs` streams into `Noise` and `Drift` by hand.

Run directly with `python -m examples.comparisons.imu.imu_nnx`.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from examples.comparisons.imu.imu_common import (
    DENSITY, DRIFT_DENSITY, DT, RES, TAU, trajectory, verify_imu,
)


class Derivative(nnx.Module):
    def __init__(self, dt: float, x0: jax.Array):
        self.dt = dt
        self.last = nnx.Variable(jnp.asarray(x0))

    def __call__(self, x: jax.Array) -> jax.Array:
        v = (x - self.last[...]) / self.dt
        self.last[...] = x
        return v


class Noise(nnx.Module):
    def __init__(self, density: float, rngs: nnx.Rngs):
        self.density = nnx.Param(jnp.asarray(density))
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.density[...] * jax.random.normal(self.rngs.noise(), jnp.shape(x))


class Drift(nnx.Module):
    def __init__(self, density: float, tau: float, dt: float, rngs: nnx.Rngs):
        self.density = nnx.Param(jnp.asarray(density))
        self.tau = nnx.Param(jnp.asarray(tau))
        self.dt = dt
        self.bias = nnx.Variable(jnp.asarray(0.0))
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        step = self.density[...] * jnp.sqrt(self.dt) * jax.random.normal(
            self.rngs.drift(), jnp.shape(x))
        self.bias[...] = self.bias[...] * (1.0 - self.dt / self.tau[...]) + step
        return x + self.bias[...]


class Quantizer(nnx.Module):
    def __init__(self, resolution: float):
        self.resolution = resolution

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.round(x / self.resolution) * self.resolution


class IMU(nnx.Module):
    def __init__(self, x0: jax.Array, rngs: nnx.Rngs,
                 density: float = DENSITY, tau: float = TAU,
                 drift_density: float = DRIFT_DENSITY):
        self.d1 = Derivative(DT, x0)
        self.d2 = Derivative(DT, jnp.asarray(0.0))
        self.noise = Noise(density, rngs)
        self.drift = Drift(drift_density, tau, DT, rngs)
        self.quant = Quantizer(RES)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.quant(self.drift(self.noise(self.d2(self.d1(x)))))


def run(positions: jax.Array, density: float = DENSITY,
        drift_density: float = DRIFT_DENSITY, seed: int = 0) -> jax.Array:
    k_noise, k_drift = jax.random.split(jax.random.PRNGKey(seed))
    imu = IMU(positions[0], nnx.Rngs(noise=k_noise, drift=k_drift),
              density=density, drift_density=drift_density)
    axes = nnx.StateAxes([
        (nnx.Param, None),
        (nnx.RngState, nnx.Carry),
        (..., nnx.Carry),
    ])

    @nnx.scan(in_axes=(axes, 0), out_axes=0)
    def over_time(model: IMU, x: jax.Array) -> jax.Array:
        return model(x)

    return over_time(imu, positions)


def main() -> None:
    _, positions = trajectory()
    pos = jnp.asarray(positions)
    accel = run(pos, seed=0)
    quiet = run(pos, density=0.0, drift_density=0.0, seed=0)
    verify_imu('nnx', pos, accel, quiet)


if __name__ == '__main__':
    main()
