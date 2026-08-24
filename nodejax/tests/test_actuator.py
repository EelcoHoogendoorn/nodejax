"""The FOC actuator stack, assembled stock and run over a spinning
mechanical state — the node-level composite port, kept runnable.

Blocks arrive constructed at the factories (bound nodes as transport
containers: stored construction values), one rng enters at init and
splits toward the sensor members, per-member state shapes derive from
the wiring by init discovery, and the stack's custom init runs the
generated walk then patches the bus estimator's boot value.
"""

import jax
import jax.numpy as jnp

from nodejax import Node, scan
from nodejax.struct import Struct
from nodejax.control import EMA, PID
from nodejax.examples.actuator import (ActuatorStack, Battery, Noisy,
                                       Encoder, Observer, torque_command,
                                       CurrentController, ModelEstimator,
                                       foc_current_model, CurrentSensor, FET,
                                       Electrical, DeratingThermal)

DT = 1e-3


def stock_stack() -> Node:
    """The full stack, member tree spelled once — at the factories."""
    return ActuatorStack(
        battery=Battery(DT)(capacity=100.0),
        mechanical_est=Encoder() >> Observer(DT),
        command_ctrl=torque_command(),
        current_ctrl=CurrentController(
            DT,
            motor=Electrical(DT)(),
            estimator=ModelEstimator(
                DT,
                filter=CurrentSensor() >> EMA(DT, warm=True)(tau=2e-3),
                model_fn=foc_current_model(DT)),
            controller=PID(DT)(kp=0.5, ki=500.0),
            fets=FET(DT)(r_th=2.0, c_th=5.0),
            bus_est=Noisy(0.2) >> EMA(DT, warm=True)(tau=0.01)),
        motor=Electrical(DT)(),
        motor_thermal=DeratingThermal(DT)(r_th=1.5, c_th=40.0, limit=100.0))


def test_stack_runs_and_tracks():
    stack = stock_stack().parameterize()
    # AT REST is a real initial condition, and supplying one is what lets the
    # warm filters prime. A resolved shape alone will not do it: zeros are a
    # spec, and a spec never enters an input slot (see test_priming)
    at_rest = Struct(mechanical=Struct(position=0.0, velocity=0.0), command=0.0)
    state = stack.init(rng=jax.random.PRNGKey(0), input=at_rest)
    # booted at the SAMPLED bus: the walk carries that condition through, so
    # the ema primes at 48 V through its own noisy sensor
    assert jnp.abs(state.current_ctrl.bus_est.ema - 48.0) < 1.0

    n = 500
    t = jnp.arange(n) * DT
    mech = Struct(position=jnp.mod(20.0 * t, 2 * jnp.pi),   # spinning at 20 rad/s
                  velocity=jnp.full(n, 20.0))
    final, torque = scan(stack)(state, mechanical=mech,
                                command=jnp.full(n, 2.0))   # 2 Nm

    assert jnp.all(jnp.isfinite(torque))
    assert jnp.abs(jnp.mean(torque[n // 2:]) - 2.0) < 0.5     # tracks the command
    assert final.battery < 1.0                                # power was drawn
    assert final.motor_thermal > 25.0                         # windings warmed
