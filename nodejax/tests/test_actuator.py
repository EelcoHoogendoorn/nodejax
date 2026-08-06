"""The FOC actuator stack, assembled stock and run over a spinning
mechanical state — the def-level composite port, kept runnable.

Blocks arrive constructed at the factories (bound nodes as transport
containers: stored construction values), one rng enters at init and
splits toward the sensor members, per-member state shapes derive from
the wiring by init discovery, and the stack's custom init runs the
generated walk then patches the bus estimator's boot value.
"""

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.examples.actuator import (actuator_stack_def, battery_def, noisy_def,
                                    ema_def, encoder_def, observer_def,
                                    torque_command_def, current_controller_def,
                                    model_estimator_def, foc_current_model,
                                    current_sensor_def, pid_def, fet_def,
                                    electrical_def, derating_thermal_def)

DT = 1e-3


def stock_stack():
    """The full stack, member tree spelled once — at the factories."""
    return actuator_stack_def(
        DT,
        battery=battery_def(DT)(capacity=100.0),
        mechanical_est=encoder_def() >> observer_def(DT),
        command_ctrl=torque_command_def(),
        current_ctrl=current_controller_def(
            DT,
            motor=electrical_def(DT)(),
            estimator=model_estimator_def(
                DT,
                filter=current_sensor_def() >> ema_def(DT)(tau=2e-3),
                model_fn=foc_current_model(DT)),
            controller=pid_def(DT)(kp=0.5, ki=500.0),
            fets=fet_def(DT)(r_th=2.0, c_th=5.0),
            bus_est=noisy_def(0.2) >> ema_def(DT)(tau=0.01)),
        motor=electrical_def(DT)(),
        motor_thermal=derating_thermal_def(DT)(r_th=1.5, c_th=40.0, limit=100.0))


def test_stack_runs_and_tracks():
    stack = stock_stack().parameterize()
    state = stack.init(rng=jax.random.PRNGKey(0))
    assert state.current_ctrl.bus_est.ema == 48.0  # booted reading the bus

    n = 500
    t = jnp.arange(n) * DT
    mech = Struct(position=jnp.mod(20.0 * t, 2 * jnp.pi),   # spinning at 20 rad/s
                  velocity=jnp.full(n, 20.0))
    stream = Struct(mechanical=mech, command=jnp.full(n, 2.0))   # 2 Nm

    final, torque = stack.scan(state, stream)

    assert jnp.all(jnp.isfinite(torque))
    assert jnp.abs(jnp.mean(torque[n // 2:]) - 2.0) < 0.5     # tracks the command
    assert final.battery < 1.0                                # power was drawn
    assert final.motor_thermal > 25.0                         # windings warmed
