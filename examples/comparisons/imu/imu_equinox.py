"""The compositional IMU, in equinox: what state costs without a contract.

The nodejax version is five nodes and one line of composition
(`imu_nodejax.py`):

    imu = derivative(DT) >> derivative(DT) >> noise() >> drift(DT) >> quantizer(RES)

with state priming, key routing, and state threading all derived from
the component definitions. This file is the same sensor in idiomatic
equinox, which is excellent at the parameter half (modules are pytrees)
and deliberately agnostic about the state half. The census of what that
agnosticism costs, at five members:

- a hand-written state container naming every member's state;
- a hand-written init composing every member's state, in order, with
  the key split count maintained by hand and the priming value
  threaded to the members that want it;
- a step function threading each member's state in and out by name, in
  the right order, rebuilding the container every step.

Every piece is idiomatic equinox and none of it is domain
physics. The recomposition drag: inserting one stateful member mid-pipe
touches the container, the init, the split count, and the step, four
places in this file, plus every caller holding the container's shape.
The nodejax diff for the same change is one term in the `>>` line.
For larger composed stateful simulations, this becomes very tedious and fragile.

Run directly:  python -m examples.comparisons.imu.imu_equinox
"""

import equinox as eqx
import jax

from nodejax import Node
import jax.numpy as jnp
import numpy as np

DT = 0.01
RES = 0.05


# --- the components: params as modules (equinox's strength) ---

class Derivative(eqx.Module):
    dt: float = eqx.field(static=True)

    def __call__(self, last, x):
        return x, (x - last) / self.dt          # (state, output), by convention


class Noise(eqx.Module):
    density: jax.Array

    def __call__(self, key, x):
        key, sub = jax.random.split(key)
        return key, x + self.density * jax.random.normal(sub)


class Drift(eqx.Module):
    density: jax.Array
    tau: jax.Array
    dt: float = eqx.field(static=True)

    def __call__(self, state, x):
        bias, key = state
        key, sub = jax.random.split(key)
        step = self.density * jnp.sqrt(self.dt) * jax.random.normal(sub)
        bias = bias * (1.0 - self.dt / self.tau) + step
        return (bias, key), x + bias


class Quantizer(eqx.Module):
    resolution: float = eqx.field(static=True)

    def __call__(self, x):
        return jnp.round(x / self.resolution) * self.resolution


# --- the state half: everything below is glue, not physics ---

class IMUState(eqx.Module):
    """One field per stateful member, maintained by hand."""
    d1_last: jax.Array
    d2_last: jax.Array
    noise_key: jax.Array
    drift: tuple


class IMU(eqx.Module):
    d1: Derivative
    d2: Derivative
    noise: Noise
    drift: Drift
    quant: Quantizer

    def init(self, x0, key):
        # compose every member's init by hand: the split count tracks the
        # number of consumers, the priming value reaches the derivatives
        k_noise, k_drift = jax.random.split(key)
        return IMUState(d1_last=jnp.asarray(x0),
                        d2_last=jnp.asarray(0.0),
                        noise_key=k_noise,
                        drift=(jnp.asarray(0.0), k_drift))

    def step(self, state, x):
        # thread each member's state in and out, in order, by name
        d1_last, v = self.d1(state.d1_last, x)
        d2_last, a = self.d2(state.d2_last, v)
        noise_key, a = self.noise(state.noise_key, a)
        drift, a = self.drift(state.drift, a)
        a = self.quant(a)
        return IMUState(d1_last=d1_last, d2_last=d2_last,
                        noise_key=noise_key, drift=drift), a


def make_imu(density: float=0.05, tau: float=1.0, drift_density: float=0.2) -> Node:
    return IMU(d1=Derivative(DT), d2=Derivative(DT),
               noise=Noise(density=jnp.asarray(density)),
               drift=Drift(density=jnp.asarray(drift_density), tau=jnp.asarray(tau), dt=DT),
               quant=Quantizer(RES))


def main() -> None:
    t = np.arange(0.0, 2.0 * np.pi, DT)
    positions = jnp.asarray(np.sin(t) + 2.0)

    imu = make_imu()
    state = imu.init(positions[0], jax.random.PRNGKey(0))
    _, accel = jax.lax.scan(lambda s, x: imu.step(s, x), state, positions)

    true_accel = np.gradient(np.gradient(np.asarray(positions), DT), DT)
    corr = np.corrcoef(np.asarray(accel)[5:], true_accel[5:])[0, 1]
    print(f"correlation with true acceleration: {corr:.3f}")
    assert corr > 0.9


if __name__ == '__main__':
    main()
