"""The actuator: the per-tick chain from battery to torque."""

from __future__ import annotations

from nodejax.struct import Struct
from nodejax import ambient, composite, composite_init


@ambient
def actuator_stack_def(dt, battery, mechanical_est, command_ctrl,
                       current_ctrl, motor, motor_thermal):
    """One control tick: battery -> voltage estimation -> command
    controller -> current controller -> electrical motor. Mechanical
    state is an INPUT (integrated at the environment level), torque is
    the output. The factory argument list is the member list; blocks
    arrive as defs or constructed (bound nodes: their params become
    the stored construction values). The current controller senses the
    bus through its own sensor pipeline ending in an ema member — its
    boot value is patched to the sampled bus.

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

    def init(ndef, param, rng):
        # the generated walk, then the boot patch: the bus estimator
        # BOOTS READING THE BUS — an EMA initialized at 0 V makes the
        # pwm normalization divide by its epsilon guard and emit
        # UNIT-NORM noise (full bus voltage in a random direction)
        # until the filter converges; a real controller samples the bus
        # before enabling the power stage
        states = composite_init(members, apply, param, input=ndef.input, rng=rng)
        # the raw unbound method spelling: init has no self in scope
        bus0 = battery.ndef.voltage(param.battery, states.battery)
        cc = states.current_ctrl
        return states.replace(current_ctrl=cc.replace(
            bus_sensor=cc.bus_sensor.replace(ema=bus0)))

    def apply(self, mechanical, command):
        true_v = self.battery.voltage()                             # method: pure read
        est_mech = self.mechanical_est(mechanical.position)         # encoder >> observer

        target = self.command_ctrl(motor=self.param.current_ctrl.motor,
                                   mechanical=est_mech, command=command)
        # the motor's thermal rollback happens HERE, where its thermal
        # lives — the current controller only derates for what it owns
        target = self.motor_thermal.derate(target)
        pwm = self.current_ctrl(current=self.state.motor,          # true electrical state
                                velocity=est_mech.velocity,
                                bus=true_v, target=target)
        out = self.motor(mechanical=mechanical, v=pwm * true_v)

        p_diss = out.current.norm2() * self.param.motor.resistance
        self.motor_thermal(p_diss)                                 # heat the windings
        self.battery(mechanical.velocity * out.torque + p_diss)
        return out.torque

    return composite(apply, members=members, init=init, name='actuator_stack',
                     apply_input_spec=Struct(mechanical=Struct(position=0.0, velocity=0.0),
                                  command=0.0))
