"""The actuator: the per-tick chain from battery to torque."""

from __future__ import annotations

from nodejax.struct import Struct
from nodejax import ambient, composite


@ambient
def ActuatorStack(dt, battery, mechanical_est, command_ctrl,
                  current_ctrl, motor, motor_thermal):
    """One control tick: battery -> voltage estimation -> command
    controller -> current controller -> electrical motor. Mechanical
    state is an INPUT (integrated at the environment level), torque is
    the output. The factory argument list is the member list; blocks
    arrive as defs or constructed (bound nodes: their params become
    the stored construction values). The current controller senses the
    bus through its own sensor pipeline ending in an ema member, and
    the init walk threads real values, so that ema BOOTS at the sampled
    bus reading (a real controller samples the bus before enabling the
    power stage; an EMA booted at 0 V would divide the pwm by its
    epsilon guard and emit unit-norm noise until it converged).

    Every estimate the controller consumes is estimated: bus voltage via
    a sensor pipeline, mechanicals via encoder >> observer (quantized,
    noisy), currents via the model-based estimator inside the current
    controller — while the physics runs on truth (motor driven by
    pwm x TRUE voltage and TRUE mechanicals). Thermals close the loop the
    other way: motor dissipation heats the motor thermal node, whose
    temperature derates the current target (the fets' own thermal lives
    inside the current controller). The battery is read via its voltage
    METHOD before the step and stepped with the drawn power after."""
    members = dict(battery=battery,
                   mechanical_est=mechanical_est, command_ctrl=command_ctrl,
                   current_ctrl=current_ctrl, motor=motor,
                   motor_thermal=motor_thermal)

    def apply(self, mechanical=Struct(position=0.0, velocity=0.0), command=0.0):
        true_v = self.battery.voltage()                             # method: pure read
        est_mech = self.mechanical_est(mechanical.position)         # encoder >> observer

        target_i = self.command_ctrl(motor=self.param.current_ctrl.motor,
                                     est_mechanical=est_mech, command=command)
        # the motor's thermal rollback happens HERE, where its thermal
        # lives — the current controller only derates for what it owns
        target_i = self.motor_thermal.derate(target_i)
        pwm = self.current_ctrl(true_i=self.state.motor,           # true electrical state
                                est_velocity=est_mech.velocity,
                                true_v=true_v, target_i=target_i)
        out = self.motor(mechanical=mechanical, v=pwm * true_v)

        p_diss = out.current.norm2() * self.param.motor.resistance
        self.motor_thermal(p_diss)                                 # heat the windings
        self.battery(mechanical.velocity * out.torque + p_diss)
        return out.torque

    return composite(apply, members=members, name='actuator_stack')
