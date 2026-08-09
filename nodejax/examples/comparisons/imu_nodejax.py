"""The compositional IMU, in nodejax: the version the other files chase.

Side by side with `imu_equinox.py` and `imu_flax.py`: the same sensor,
with the state container, init composition, threading and key routing
derived from the component definitions.

position -> derivative >> derivative >> noise >> drift >> quantizer -> accel

Everything this pipeline needs is existing machinery, composed:
- derivative state PRIMES from the first sample (the input channel) — no spike
- mid-pipe streaming randomness is rng-as-state: auto-advanced keys, routed
  to the stochastic members by composite init from ONE boundary key
- statics are closures, mixed bound/unbound members promote, and the whole
  pipe is one cyclic node that plugs straight into scan
"""

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
import numpy as np

from nodejax import NodeDef

from nodejax import node_def

DT = 0.01
RES = 0.05


def Derivative(dt):
    """Discrete derivative. Its state (the previous sample) PRIMES from the
    init input value — zero is a poor default; the first real sample
    is the right one. With priming, the
    first output is 0 instead of a (x0 - 0)/dt spike."""
    def init(input):
        return jnp.asarray(input)          # DATA: primes from the real first sample
    def apply(state, input):
        return input, (input - state) / dt
    return node_def(apply, init=init, name='derivative')


def Noise():
    """Additive white noise; density is a param (trainable, e.g. for sensor
    model fitting). Streaming randomness = rng-as-state: the reserved rng
    field auto-advances, and composite init routes a key here mid-pipe."""
    def param(density):
        return Struct(density=jnp.asarray(density))
    def init(param, rng):
        return Struct(rng=rng)
    def apply(param, state, input):
        return state, input + param.density * jax.random.normal(state.rng)
    return node_def(apply, param=param, init=init, name='noise')


def Drift(dt):
    """Slowly wandering bias (Ornstein-Uhlenbeck-ish): cyclic state carries
    both the bias and its noise stream."""
    def param(density, tau):
        return Struct(density=jnp.asarray(density), tau=jnp.asarray(tau))
    def init(param, rng):
        return Struct(bias=jnp.asarray(0.0), rng=rng)
    def apply(param, state, input):
        step = param.density * jnp.sqrt(dt) * jax.random.normal(state.rng)
        bias = state.bias * (1.0 - dt / param.tau) + step
        return state.replace(bias=bias), input + bias
    return node_def(apply, param=param, init=init, name='drift')


def quantizer(resolution):
    """Round to the sensor's resolution grid; plain stateless node."""
    return node_def(lambda input: jnp.round(input / resolution) * resolution,
                    name='quantizer')



def imu_pipe():
    return (Derivative(DT) >> Derivative(DT)
            >> Noise() >> Drift(DT) >> quantizer(RES))


def make_imu(density=0.05, tau=1.0, drift_density=0.2):
    pipe = imu_pipe()
    assert isinstance(pipe, NodeDef) and pipe.parametric and pipe.cyclic
    return pipe.parameterize(
        noise=Struct(density=jnp.asarray(density)),
        drift=Struct(tau=jnp.asarray(tau), density=jnp.asarray(drift_density)),
    )


def trajectory():
    t = np.arange(0.0, 2.0 * np.pi, DT)
    return t, jnp.asarray(np.sin(t) + 2.0)  # offset makes priming observable


def run(imu, positions, seed=0):
    state = imu.init(input=positions[0], rng=jax.random.PRNGKey(seed))
    _, accel = imu.scan(state, positions)
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

    a0 = run(imu, positions, seed=0)
    a0_again = run(imu, positions, seed=0)
    a1 = run(imu, positions, seed=1)
    assert jnp.allclose(a0, a0_again)          # same key, same sensor tape
    assert not jnp.allclose(a0, a1)            # different key, different tape

    state = imu.init(input=positions[0], rng=jax.random.PRNGKey(0))
    assert jnp.any(state.noise.rng != state.drift.rng)  # split, not copied


def test_quiet_imu_recovers_exact_dynamics():
    """With noise/drift silenced the pipeline is a pure discrete second
    derivative — checked against the closed form."""
    _, positions = trajectory()
    imu = imu_pipe().parameterize(
        noise=Struct(density=jnp.asarray(0.0)),
        drift=Struct(tau=jnp.asarray(1.0), density=jnp.asarray(0.0)),
    )
    accel = run(imu, positions)

    p = np.asarray(positions)
    exact = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / DT ** 2   # discrete 2nd difference
    quantized = np.round(exact / RES) * RES
    assert jnp.allclose(accel[2:], quantized, atol=1e-4)
