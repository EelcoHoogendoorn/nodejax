"""Block-level tests for the actuator package: the pipeline PID, DQ
signal polymorphism, and the monolithic full-fidelity plant (cogging,
stiction, hysteresis — the physics the split emotor/mechanical pair
deliberately omits)."""

import jax
import jax.numpy as jnp

from nodejax import scan
from nodejax.struct import Struct
from nodejax.examples.actuator import (DQ, motor_params, plant_def,
                                    pid_def, rate_limit_def, clamp_def, wrap_def)

DT = 1e-4


def test_pid_is_a_pipeline():
    """Optional PID behavior is a pipe stage you include or don't — and
    each stage is independently a node (the rate limiter carries its own
    state)."""
    bare = pid_def(DT).parameterize(kp=2.0, ki=0.0)
    assert bare.apply(bare.with_input(jnp.asarray(0.0)).init(), 1.0)[1] == 2.0

    full = (wrap_def() >> pid_def(DT) >> rate_limit_def(DT) >> clamp_def()).parameterize(
        pid=Struct(kp=2.0, ki=0.0),
        rate_limit=Struct(max_rate=5000.0),        # units per second
        clamp=Struct(limit=1.5))
    s = full.with_input(jnp.asarray(0.0)).init()
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 0.5)               # slew-limited: 5000 * dt
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)               # clamp takes over
    s, out = full.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)


def test_pid_core_is_signal_polymorphic():
    """The SAME pid def runs scalars or DQ pairs: its state derives from
    the init input value — signal type is decided at init, not by
    subclassing."""
    pid = pid_def(DT).parameterize(kp=2.0, ki=0.0)

    s = pid.with_input(DQ()).init()
    assert isinstance(s.integral, DQ)

    s, out = pid.apply(s, DQ(1.0, -2.0))
    assert out.d == 2.0 and out.q == -4.0


def test_stiction_holds_the_rotor():
    """Plant fidelity: below the static-torque threshold the rotor sticks;
    zero the threshold and the same drive spins it."""
    drive = Struct(v=DQ(jnp.zeros(400), jnp.full(400, 0.1)), load=jnp.zeros(400))

    sticky = scan(plant_def(DT)).bind(motor_params(torque_static=2.0))
    assert jnp.all(sticky.apply(drive).velocity == 0.0)

    free = scan(plant_def(DT)).bind(motor_params(torque_static=0.0))
    assert free.apply(drive).velocity[-1] > 0.0


def test_recorded_rollout():
    """scan(record=True): the full state trajectory rides along — the
    Simulation.run convention as a stock transform (keyless plants)."""
    seq = scan(plant_def(DT), record=True).bind(motor_params())
    drive = Struct(v=DQ(jnp.zeros(300), jnp.full(300, 1.0)), load=jnp.zeros(300))
    ys = seq.apply(drive)

    assert ys.state.velocity.shape == (300,)
    assert ys.output.position.shape == (300,)
    assert ys.state.velocity[-1] > 0.0
