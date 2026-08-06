"""The compositional IMU, in flax nnx: mutation makes the step easy and
the boundary hard.

The nodejax version is five defs and one line of composition
(`nodejax/tests/test_imu.py`); the equinox counterpart
(`imu_equinox.py`) pays for statelessness with a hand-written state
container, init, and threading step. nnx sits at the other pole:
modules are mutable Python objects, so the per-step wiring below is
the cleanest of the three, each member updating its own variables in
place. The census of where the cost moved:

- construction composes state by hand anyway: each member's __init__
  seeds its variables, the priming value and the rng streams are
  plumbed through the constructors, and the composite constructor is
  the init function, just spelled as object construction;
- the mutable module cannot cross a jax boundary as itself: running
  under `lax.scan` means `nnx.split` into (graphdef, state), a step
  that merges, calls, and re-splits, and a final merge to get the
  object back. The threading equinox writes per member, nnx writes
  per boundary;
- inserting a stateful member touches two places (its construction,
  its call site), better than equinox's four, with the split/merge
  ceremony unchanged.

Every piece is idiomatic nnx. The three files side by side are the
comparison: where each framework puts the state work that nodejax
derives from the contract.

Run directly:  python -m nodejax.examples.comparisons.imu_flax
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
        # mutation threads the state: the cleanest step of the three files
        return self.quant(self.drift(self.noise(self.d2(self.d1(x)))))


def main():
    t = np.arange(0.0, 2.0 * np.pi, DT)
    positions = jnp.asarray(np.sin(t) + 2.0)

    imu = IMU(positions[0], nnx.Rngs(noise=jax.random.PRNGKey(0),
                                     drift=jax.random.PRNGKey(1)))

    # the boundary ceremony: a mutable module cannot ride lax.scan as
    # itself; split to values, merge-call-split per step, merge at the end
    graphdef, state = nnx.split(imu)

    def step(state, x):
        module = nnx.merge(graphdef, state)
        y = module(x)
        _, state = nnx.split(module)
        return state, y

    state, accel = jax.lax.scan(step, state, positions)
    imu = nnx.merge(graphdef, state)

    true_accel = np.gradient(np.gradient(np.asarray(positions), DT), DT)
    corr = np.corrcoef(np.asarray(accel)[5:], true_accel[5:])[0, 1]
    print(f"correlation with true acceleration: {corr:.3f}")
    assert corr > 0.9


if __name__ == '__main__':
    main()
