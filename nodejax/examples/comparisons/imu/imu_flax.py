"""The compositional IMU, in Flax NNX: graph-aware transforms over objects.

The NodeJAX version is five nodes and one line of composition
(`imu_nodejax.py`). The Equinox counterpart (`imu_equinox.py`) uses
explicit functional state. NNX takes a different approach: modules are
mutable Python objects, and its graph-aware scan carries their Variables
while preserving the object graph. The result has two distinct parts:

- Construction routes the priming value and named RNG streams through the
  object constructors.
- The step is ordinary object composition. Each member updates its own
  Variables in place.
- `nnx.scan` accepts the module directly. `StateAxes` declares that Params
  are shared across time while all other Variables carry from step to step.

The three files compare where each framework records the same state
information. This file uses the current NNX transform boundary rather than
manually lowering the module with `split` and `merge`.

Run directly:  python -m nodejax.examples.comparisons.imu.imu_flax
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

DT = 0.01
RES = 0.05


class Derivative(nnx.Module):
    def __init__(self, dt, x0):
        self.dt = dt
        self.last = nnx.Variable(jnp.asarray(x0))   # primed by the constructor

    def __call__(self, x):
        v = (x - self.last[...]) / self.dt
        self.last[...] = x
        return v


class Noise(nnx.Module):
    def __init__(self, density, rngs: nnx.Rngs):
        self.density = nnx.Param(jnp.asarray(density))
        self.rngs = rngs

    def __call__(self, x):
        return x + self.density[...] * jax.random.normal(self.rngs.noise())


class Drift(nnx.Module):
    def __init__(self, density, tau, dt, rngs: nnx.Rngs):
        self.density = nnx.Param(jnp.asarray(density))
        self.tau = nnx.Param(jnp.asarray(tau))
        self.dt = dt
        self.bias = nnx.Variable(jnp.asarray(0.0))
        self.rngs = rngs

    def __call__(self, x):
        step = self.density[...] * jnp.sqrt(self.dt) * jax.random.normal(self.rngs.drift())
        self.bias[...] = self.bias[...] * (1.0 - self.dt / self.tau[...]) + step
        return x + self.bias[...]


class Quantizer(nnx.Module):
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, x):
        return jnp.round(x / self.resolution) * self.resolution


class IMU(nnx.Module):
    def __init__(self, x0, rngs: nnx.Rngs,
                 density=0.05, tau=1.0, drift_density=0.2):
        # object construction IS the composed init, written by hand:
        # priming value to the derivatives, rng streams to the consumers
        self.d1 = Derivative(DT, x0)
        self.d2 = Derivative(DT, 0.0)
        self.noise = Noise(density, rngs)
        self.drift = Drift(drift_density, tau, DT, rngs)
        self.quant = Quantizer(RES)

    def __call__(self, x):
        # mutation threads the state: the cleanest step of the three files.
        # nnx.Sequential could fold this line, but it composes CALLS only:
        # the constructor plumbing above (priming value, rng routing) is
        # the part a serial combinator would need to compose, and does not
        return self.quant(self.drift(self.noise(self.d2(self.d1(x)))))


def main() -> None:
    t = np.arange(0.0, 2.0 * np.pi, DT)
    positions = jnp.asarray(np.sin(t) + 2.0)

    imu = IMU(positions[0], nnx.Rngs(noise=jax.random.PRNGKey(0),
                                     drift=jax.random.PRNGKey(1)))

    axes = nnx.StateAxes([
        (nnx.Param, None),
        (nnx.RngState, nnx.Carry),
        (..., nnx.Carry),
    ])

    @nnx.scan(in_axes=(axes, 0), out_axes=0)
    def over_time(model, input):
        return model(input)

    accel = over_time(imu, positions)
    print(nnx.state(imu))

    true_accel = np.gradient(np.gradient(np.asarray(positions), DT), DT)
    corr = np.corrcoef(np.asarray(accel)[5:], true_accel[5:])[0, 1]
    print(f"correlation with true acceleration: {corr:.3f}")
    assert corr > 0.9


if __name__ == '__main__':
    main()
