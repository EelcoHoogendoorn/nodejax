"""Defines `CurrentController`, per-term voltage `Feedforward`, and outer torque or velocity command Nodes."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import BaseNode, Node, PNode, PyTree, node, Leaf, Composite

from examples.actuator.dq import DQ
from nodejax.control import ClampNorm, Delay, Diff
from examples.actuator.motor import current_model


@node
def foc_current_model(dt: float) -> Callable[[DQ, DQ, Struct], DQ]:
    """Returns a current-prediction function using the controller's estimated bus voltage, velocity, and motor parameters."""
    def model_fn(filtered: DQ, previous: DQ, model: Struct) -> DQ:
        di_dt = (filtered - previous) / dt
        return current_model(model.motor, model.mod_v, di_dt, model.est_velocity)
    return model_fn


@node(name='ff')
def Feedforward() -> Node:
    """Sums resistive, back-EMF, and inductive voltage terms weighted by per-term gains `(r, bemf, l)`."""
    def param(r: PyTree, bemf: PyTree, l: PyTree) -> Struct:
        return Struct(r=r, bemf=bemf, l=l)

    def apply(param, resistive: DQ, bemf: DQ, inductive: DQ) -> DQ:
        return (resistive * param.r + bemf * param.bemf
                + inductive * param.l)

    return Leaf(apply, param=param)


@node
def CurrentController(
        dt: float, motor: BaseNode, estimator: BaseNode,
        controller: BaseNode, fets: BaseNode, bus_est: BaseNode,
) -> Node | PNode:
    """Closes the `DQ`-axis current loop and outputs a norm-clamped PWM duty vector.

    Senses `true_v` through `bus_est`, clamps `target_i` by norm (`limit`) and power-stage temperature (`fets.derate`), estimates `DQ` current via `estimator` using the previous commanded PWM duty and `est_v`, adds feedback (`controller`) to reference-derivative voltage feedforward (`motor.voltage_terms` weighted by `ff`), divides by `max(est_v, 1e-3)`, and steps `fets` and `pwm_prev`.
    """
    members = Composite(motor=motor, estimator=estimator, controller=controller,
                        fets=fets, bus_est=bus_est,
                        ff=Feedforward(), limit=ClampNorm(),
                        pwm_prev=Delay().with_input(DQ()), d_dt=Diff(dt))

    def apply(self, true_i: DQ, est_velocity, true_v, target_i: DQ) -> DQ:
        est_v = self.bus_est(true_v)
        target_i = self.limit(target_i)
        target_i = self.fets.derate(target_i)

        # Predict current from the controller's previous PWM command and estimated bus voltage.
        est_i = self.estimator(
            value=true_i,
            model=Struct(motor=self.motor.param,
                         mod_v=self.pwm_prev.state * est_v,
                         est_velocity=est_velocity))

        fb_v = self.controller(target_i - est_i)
        # Drive inductive feedforward from the reference derivative d(target_i)/dt rather than noisy current error.
        terms = self.motor.voltage_terms(target_i, self.d_dt(target_i), est_velocity)
        voltage = fb_v + self.ff(bundle=terms)

        pwm = (voltage / jnp.maximum(est_v, 1e-3)).clamp_norm(1.0)

        self.fets(est_i.norm2())                  # fets own r_dson: amps^2 in
        self.pwm_prev(pwm)
        return pwm

    return members(apply)


@node
def torque_command() -> PNode:
    """Converts a scalar shaft-torque command into a quadrature-only `DQ(d=0, q=command / motor.kt)` current target."""
    def apply(command, motor: Struct) -> DQ:
        iq = command / motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)
    return Leaf(apply)


@node
def VelocityCommand(velocity_ctrl: BaseNode) -> Node | PNode:
    """Maps a velocity setpoint and estimated mechanical state to a quadrature `DQ` current target via `velocity_ctrl`."""
    members = Composite(velocity_ctrl=velocity_ctrl)

    def apply(self, command, est_mechanical: Struct, motor: Struct) -> DQ:
        torque = self.velocity_ctrl(command - est_mechanical.velocity)
        iq = torque / motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)

    return members(apply)
