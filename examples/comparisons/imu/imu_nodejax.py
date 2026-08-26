"""The compositional IMU, in nodejax: the version the other files chase.

Side by side with `imu_equinox.py` and `imu_flax.py`: the same sensor,
with the state container, init composition, threading and key routing
derived from the component definitions.

position -> derivative >> derivative >> noise >> drift >> quantizer -> accel

Everything this pipeline needs is existing machinery, composed:
- derivative state PRIMES from the first sample (the input slot) — no spike
- mid-pipe randomness is rng-as-state: auto-advanced keys, routed
  to the stochastic members by composite init from ONE boundary key
- statics are closures, mixed bound/unbound members promote, and the whole
  pipe is one cyclic node that plugs straight into scan
"""

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
import numpy as np

from nodejax import node, scan, Node

from nodejax import Leaf, residual, sum_junction
from nodejax.control import Integrator, Quantize

DT = 0.01
RES = 0.05


@node
def Derivative(dt: float) -> Node:
    """Discrete derivative. Its state (the previous sample) PRIMES from the
    init input value — zero is a poor default; the first real sample
    is the right one. With priming, the
    first output is 0 instead of a (x0 - 0)/dt spike."""
    def init(input):
        return jnp.asarray(input)          # DATA: primes from the real first sample
    def apply(state, input):
        return input, (input - state) / dt
    return Leaf(apply, init=init)


@node
def Noise() -> Node:
    """White noise: a DRAW, not a filter. Density is a param (trainable,
    e.g. for sensor model fitting). Streaming randomness is rng-as-state: the
    state field auto-advances, while composite init routes its separate call
    frame here mid-pipe.

    It returns the noise rather than the noisy signal, so the addition
    lives in the summing junction where it is visible, instead of hiding
    inside a stage that pretends to transform. The input is still what
    gives the draw its SHAPE; only its value goes unused."""
    def param(density):
        return Struct(density=jnp.asarray(density))

    def init(param, rng):
        return Struct(rng=rng)

    def apply(param, state, input):
        return state, param.density * jax.random.normal(state.rng, jnp.shape(input))

    return Leaf(apply, param=param, init=init)


def make_imu(density: float=0.05, tau: float=1.0, drift_density: float=0.2) -> Node:
    """position -> d/dt -> d/dt -> (+ noise + drift) -> quantize -> accel.

    The disturbances are CONTRIBUTIONS, not stages: the summing junction adds
    what each makes of the same signal, and residual puts the signal itself
    back in. Spelled as a flat pipe instead, every disturbance would perform
    its own hidden addition and the topology would be a lie.

    Drift needs no node of its own. A wandering bias IS noise into a leaky
    integrator, so it is written as one, and sqrt(dt) rides on the density
    where the draw is made rather than as a gain somewhere downstream.

    Each block is parameterized where it is BUILT, so nothing has to address
    anything through the tree; the bound members promote into the unbound
    pipe around them."""
    drift = Noise()(density=drift_density * jnp.sqrt(DT)) >> Integrator()(decay=DT / tau)
    return (Derivative(DT) >> Derivative(DT)
            >> residual(sum_junction(noise=Noise()(density=density), drift=drift))
            >> Quantize(RES))


def trajectory():
    t = np.arange(0.0, 2.0 * np.pi, DT)
    return t, jnp.asarray(np.sin(t) + 2.0)  # offset makes priming observable


def run(imu: Node, positions, key: jax.Array=0):
    sensor = imu.with_input(positions[0]).parameterize().initialize(
        input=positions[0], rng=jax.random.PRNGKey(key))
    _, accel = sensor.scan(positions)
    return accel


def test_imu_measures_acceleration():
    """The simulated IMU tracks the true second derivative through two
    derivations, noise, drift and quantization."""
    t, positions = trajectory()
    accel = run(make_imu(), positions)

    true_accel = np.gradient(np.gradient(np.asarray(positions), DT), DT)
    # ignore the first samples where the primed derivatives still settle
    corr = np.corrcoef(np.asarray(accel)[5:], true_accel[5:])[0, 1]
    assert corr > 0.9


def test_derivative_priming_prevents_spike():
    """State primed from the first REAL sample (position offset 2.0): the
    first output stays small, where a zero-initialized previous-sample
    would produce a (2/dt)/dt ~ 2e4 transient."""
    _, positions = trajectory()
    accel = run(make_imu(), positions)
    assert jnp.abs(accel[0]) < 1.0


def test_output_is_on_the_quantization_grid():
    _, positions = trajectory()
    accel = run(make_imu(), positions)
    assert jnp.allclose(accel, jnp.round(accel / RES) * RES, atol=1e-5)


def test_noise_streams_are_keyed_and_pure():
    """One boundary key -> independent streams for noise and drift members;
    trajectories are pure functions of that key."""
    _, positions = trajectory()
    imu = make_imu()

    a0 = run(imu, positions, key=0)
    a0_again = run(imu, positions, key=0)
    a1 = run(imu, positions, key=1)
    assert jnp.allclose(a0, a0_again)          # same key, same sensor tape
    assert not jnp.allclose(a0, a1)            # different key, different tape

    sensor = imu.with_input(positions[0]).parameterize().initialize(
        input=positions[0], rng=jax.random.PRNGKey(0))
    dist = sensor.state.res_noise_drift
    assert jnp.any(dist.noise.rng != dist.drift.noise.rng)  # split, not copied


def test_quiet_imu_recovers_exact_dynamics():
    """With noise/drift silenced the pipeline is a pure discrete second
    derivative — checked against the closed form."""
    _, positions = trajectory()
    imu = make_imu(density=0.0, drift_density=0.0)
    accel = run(imu, positions)

    p = np.asarray(positions)
    exact = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / DT ** 2   # discrete 2nd difference
    quantized = np.round(exact / RES) * RES
    assert jnp.allclose(accel[2:], quantized, atol=1e-4)
