"""The actuator: the per-tick chain from bus voltage to torque."""

from nodejax.struct import Struct
from nodejax import Node, node, ambient, Composite


@node
def ActuatorStack(mechanical_est: Node, command_ctrl: Node,
                  current_ctrl: Node, motor: Node, motor_thermal: Node) -> Node:
    """One control tick: bus voltage -> voltage estimation -> command
    controller -> current controller -> electrical motor. Mechanical
    state and the true bus voltage are INPUTS, integrated and supplied at
    the environment level, where the pack that feeds one or several of
    these stacks lives; the output is the torque and the power drawn from
    the bus, shaft power plus dissipation, for that pack to account. The
    factory argument list is the member list; blocks arrive as nodes or
    constructed (bound nodes: their params become the stored construction
    values). The current controller senses the bus through its own sensor
    pipeline ending in an ema member, and the init walk threads real
    values, so that ema BOOTS at the sampled bus reading (a real controller
    samples the bus before enabling the power stage; an EMA booted at 0 V
    would divide the pwm by its epsilon guard and emit unit-norm noise
    until it converged).
    """
    members = Composite(mechanical_est=mechanical_est, command_ctrl=command_ctrl,
                   current_ctrl=current_ctrl, motor=motor,
                   motor_thermal=motor_thermal)
    def apply(self, mechanical, command, bus_voltage):
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
        """What the controller knows of the mechanism: its observer's
        position and velocity estimate and its estimated quadrature current.
        The readout a policy above the stack may see; the truth is not it."""
        return Struct(
            angle=state.mechanical_est.observer.position,
            velocity=state.mechanical_est.observer.velocity,
            current=state.current_ctrl.estimator.prev.q,
        )

    return members(apply, methods={'observe': observe})
