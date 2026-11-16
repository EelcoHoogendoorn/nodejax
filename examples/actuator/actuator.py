"""Defines `ActuatorStack`, the per-tick control and electrical-physics Composite from bus voltage to shaft torque."""

from __future__ import annotations

from nodejax.struct import Struct
from nodejax import BaseNode, Node, PNode, node, Composite


@node
def ActuatorStack(
        mechanical_est: BaseNode, command_ctrl: BaseNode,
        current_ctrl: BaseNode, motor: BaseNode, motor_thermal: BaseNode,
) -> Node | PNode:
    """Runs one FOC control and electrical motor tick from `(mechanical, command, bus_voltage)` to `Struct(torque, power)`.

    Threads `mechanical.position` through `mechanical_est`, computes desired `DQ` current via `command_ctrl`, derates that target through `motor_thermal`, closes the current loop in `current_ctrl`, steps `motor`, and feeds resistive winding loss into `motor_thermal`. Warm-initializes the bus-voltage `EMA` inside `current_ctrl` from concrete initial `bus_voltage` values so PWM duty scaling starts at the sampled bus reading rather than zero.
    """
    members = Composite(mechanical_est=mechanical_est, command_ctrl=command_ctrl,
                        current_ctrl=current_ctrl, motor=motor,
                        motor_thermal=motor_thermal)

    def apply(self, mechanical: Struct, command, bus_voltage) -> Struct:
        true_v = bus_voltage
        est_mech = self.mechanical_est(mechanical.position)         # encoder >> observer

        target_i = self.command_ctrl(bundle=Struct(
            command=command,
            est_mechanical=est_mech,
            motor=self.current_ctrl.param.motor))
        target_i = self.motor_thermal.derate(target_i)
        pwm = self.current_ctrl(true_i=self.motor.state,           # true electrical state
                                est_velocity=est_mech.velocity,
                                true_v=true_v, target_i=target_i)
        out = self.motor(mechanical=mechanical, voltage=pwm * true_v)

        p_diss = out.current.norm2() * self.motor.param.resistance
        self.motor_thermal(p_diss)                                 # heat the windings
        return Struct(torque=out.torque, power=mechanical.velocity * out.torque + p_diss)

    def observe(state) -> Struct:
        """Extracts estimated rotor angle, velocity, and quadrature current from `state`."""
        return Struct(
            angle=state.mechanical_est.observer.position,
            velocity=state.mechanical_est.observer.velocity,
            current=state.current_ctrl.estimator.prev.q,
        )

    return members(apply, methods={'observe': observe})
