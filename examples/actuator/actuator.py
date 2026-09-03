"""The actuator: the per-tick chain from battery to torque."""

from nodejax.struct import Struct
from nodejax import Node, node, ambient, Composite


@node
def ActuatorStack(battery: Node, mechanical_est: Node, command_ctrl: Node,
                  current_ctrl: Node, motor: Node, motor_thermal: Node) -> Node:
    """One control tick: battery -> voltage estimation -> command
    controller -> current controller -> electrical motor. Mechanical
    state is an INPUT (integrated at the environment level), torque is
    the output. The factory argument list is the member list; blocks
    arrive as nodes or constructed (bound nodes: their params become
    the stored construction values). The current controller senses the
    bus through its own sensor pipeline ending in an ema member, and
    the init walk threads real values, so that ema BOOTS at the sampled
    bus reading (a real controller samples the bus before enabling the
    power stage; an EMA booted at 0 V would divide the pwm by its
    epsilon guard and emit unit-norm noise until it converged).
    """
    members = Composite(battery=battery,
                   mechanical_est=mechanical_est, command_ctrl=command_ctrl,
                   current_ctrl=current_ctrl, motor=motor,
                   motor_thermal=motor_thermal)
    def apply(self, mechanical, command):
        true_v = self.battery.voltage()                             # method: pure read
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
        self.battery(mechanical.velocity * out.torque + p_diss)
        return out.torque

    return members(apply)
