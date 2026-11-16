"""Verifies individual actuator `Node`s including sensor calibration, `FET` dissipation, pipeline `PID` stages, `DQ` signal polymorphism, and `BenchMotor` stiction."""

import jax
import jax.numpy as jnp

from nodejax import scan, scanned
from nodejax.struct import Struct
from nodejax.control import PID, RateLimit, Clamp
from examples.actuator import (DQ, BenchMotor, CurrentSensor, Encoder,
                                       FET, Noisy, wrap)

DT = 1e-4
BENCH_ELECTRICAL = Struct(
    resistance=0.24,
    inductance_d=2e-4,
    inductance_q=3e-4,
    kt=1.2,
    pole_pairs=16.0,
    slots=36.0,
    cogging=0.8,
)


def test_sensor_calibration_is_parameter_data() -> None:
    encoder = Encoder().parameterize(
        resolution=4096.0, noise_std=0.2, phase_offset=0.3)
    current = CurrentSensor().parameterize(noise_std=0.4)
    voltage = Noisy().parameterize(noise_std=0.5)

    assert encoder.statics_by_path() == {}
    assert current.statics_by_path() == {}
    assert voltage.statics_by_path() == {}
    assert jnp.allclose(encoder.param.resolution, 4096.0)
    assert jnp.allclose(encoder.param.noise_std, 0.2)
    assert jnp.allclose(encoder.param.phase_offset, 0.3)
    assert jnp.allclose(current.param.noise_std, 0.4)
    assert jnp.allclose(voltage.param.noise_std, 0.5)


def test_fet_derivation_keeps_flat_params_and_dispatches_dissipation():
    fet = FET(DT).parameterize(
        r_th=2.0,
        c_th=5.0,
        ambient=25.0,
        limit=80.0,
        hardness=4.0,
        r_dson=0.02,
    )

    assert fet.param.__keys__ == (
        'r_th', 'c_th', 'ambient',
        'limit', 'hardness', 'r_dson',
    )
    assert jnp.allclose(fet.dissipation(100.0), 2.0)

    state = fet.init()
    next_state, output = fet.apply(state, 100.0)
    expected = 25.0 + 2.0 / 5.0 * DT
    assert jnp.allclose(next_state, expected)
    assert jnp.allclose(output, expected)


def test_pid_is_a_pipeline():
    """Verifies that rate limiting and clamping compose with `PID` as independent `Node` stages."""
    bare = PID(DT).with_input(jnp.asarray(0.0)).parameterize(
        kp=2.0, ki=0.0)
    assert bare.apply(bare.init(), 1.0)[1] == 2.0

    full = (wrap() >> PID(DT) >> RateLimit(DT) >> Clamp()).with_input(
        jnp.asarray(0.0)).parameterize(
            pid=Struct(kp=2.0, ki=0.0),
            rate_limit=Struct(max_rate=5000.0),
            clamp=Struct(limit=1.5))
    s = full.init()
    # Rate limiting caps the first step to 5000 * dt = 0.5, and clamping caps later steps at 1.5.
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 0.5)
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)


def test_pid_core_is_signal_polymorphic():
    """Verifies that `PID` derives `DQ` integrator state directly from a `DQ` input prototype."""
    pid = PID(DT).with_input(DQ()).parameterize(kp=2.0, ki=0.0)

    s = pid.init()
    assert type(s.integral) is DQ

    s, out = pid.apply(s, DQ(1.0, -2.0))
    assert out.d == 2.0 and out.q == -4.0


def test_stiction_holds_the_rotor():
    """Verifies that `BenchMotor` remains stationary below its static-torque threshold and spins when that threshold is zero."""
    drive = Struct(voltage=DQ(jnp.zeros(400), jnp.full(400, 0.1)), load=jnp.zeros(400))

    def bench(torque_static):
        return scanned(BenchMotor(DT)).parameterize(
            electrical=BENCH_ELECTRICAL,
            mechanical=Struct(inertia=0.1, friction=1e-3,
                               torque_static=torque_static))

    assert jnp.all(bench(2.0).apply(bundle=drive).velocity == 0.0)
    assert bench(0.0).apply(bundle=drive).velocity[-1] > 0.0


def test_recorded_rollout():
    """Verifies that `scanned(..., record=True)` preserves the primary output trajectory while emitting per-step states on `Aux`."""
    seq = scanned(BenchMotor(DT), record=True).parameterize(
        electrical=BENCH_ELECTRICAL,
        mechanical=Struct(inertia=0.1, friction=1e-3))
    outs, aux = seq.apply(voltage=DQ(jnp.zeros(300), jnp.full(300, 1.0)),
                          load=jnp.zeros(300))

    assert outs.position.shape == (300,)
    assert aux.state.mechanical.velocity.shape == (300,)
    assert aux.state.mechanical.velocity[-1] > 0.0
