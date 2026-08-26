"""Current controller: target current + measurements in, PWM out.

Composes a model-based estimator, one DQ PID, the motor member (the
controller's model of the plant: the same motor def the plant runs,
here consulted through its methods and params), the power
stage's own thermal (fets), the per-term feedforward weighting (ff) and
a norm-clamp current limiter (limit), a one-tick memory (previous pwm,
feeding the estimator's model) and a difference node (d_dt, the
reference derivative driving the inductive feedforward). Output is a PWM command —
voltage normalized by the ESTIMATED bus voltage, norm-clamped to 1.
The fets heat on their own conduction losses (read-then-step) and
derate the target; derating for components this controller does NOT
own (the motor windings) happens upstream, where those thermals live.
"""

from __future__ import annotations

import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, node, ambient, Leaf, Composite

from examples.actuator.dq import DQ
from nodejax.control import ClampNorm, Delay, Diff
from examples.actuator.motor import current_model, voltage_terms


@node
def foc_current_model(dt: float):
    """Predict the current the previously commanded modulated voltage
    implies, with di/dt taken filtered-vs-previous — the physics as
    plain functions on plain param data. The model bundle carries:
    motor (the model params), mod_v (the controller's account of the
    voltage applied last step: its commanded pwm times its bus
    estimate), est_velocity (the estimated mechanical velocity) — every
    model input is an estimate; only the plant sees truth."""
    def model_fn(filtered, previous, model):
        di_dt = (filtered - previous) / dt
        return current_model(model.motor, model.mod_v, di_dt, model.est_velocity)
    return model_fn


@node(name='ff')
def Feedforward() -> Node:
    """Per-term feedforward: the model's voltage terms weighted by the
    trust placed in each (resistive/bemf/inductive gains), summed. A
    leaf node — the terms its input, the gains its params."""
    def param(r, bemf, l):
        return Struct(r=r, bemf=bemf, l=l)

    def apply(param, resistive, bemf, inductive):
        return (resistive * param.r + bemf * param.bemf
                + inductive * param.l)

    return Leaf(apply, param=param)


@node
def CurrentController(dt: float, motor: Node, estimator: Node, controller: Node, fets: Node, bus_est: Node) -> Node:
    """Target current + measurements in, PWM out.

    input fields, declared by the apply signature: true_i (true DQ
    current), est_velocity (the observer's estimated mechanical
    velocity; every velocity this controller sees is an estimate),
    true_v (the true bus voltage; the controller senses it through its
    own bus_est member, as it senses current through the estimator's),
    target_i (DQ current target). The target arrives PRE-DERATED for anything this controller does
    not own (the motor's thermal rollback happens where the motor
    thermal lives — at the stack). The factory argument list is the
    member list; ff (per-term feedforward weighting) and limit (a
    norm-clamp on the target current) are leaf-node members like the
    rest."""
    members = Composite(motor=motor, estimator=estimator, controller=controller,
                   fets=fets, bus_est=bus_est,
                   ff=Feedforward(), limit=ClampNorm(),
                   pwm_prev=Delay().with_input(DQ()), d_dt=Diff(dt))

    def apply(self, true_i, est_velocity, true_v, target_i):
        # the controller owns its ADCs: the TRUE bus voltage arrives and
        # is sensed here, exactly as the current is sensed inside the
        # estimator member; everything past this line is an estimate —
        # physics runs real, the controller runs on estimations
        est_v = self.bus_est(true_v)
        target_i = self.limit(target_i)
        # the power stage derates by its own temperature (read-then-step)
        target_i = self.fets.derate(target_i)

        # model-based estimation: the model sees the controller's own
        # account of the voltage applied last step (its commanded pwm
        # times its bus estimate — the true bus is not its to know)
        est_i = self.estimator(
            value=true_i,
            model=Struct(motor=self.param.motor,
                         mod_v=self.state.pwm_prev * est_v,
                         est_velocity=est_velocity))

        fb_v = self.controller(target_i - est_i)
        # feedforward with PER-TERM trust weights: resistive and
        # speed-voltage terms at the target operating point, and the
        # inductive term driven by the REFERENCE derivative — an
        # error-driven di/dt is a hidden P-gain of L/dt on measurement
        # noise (measured: tracking flat, jitter x4)
        terms = self.motor.voltage_terms(target_i, self.d_dt(target_i), est_velocity)
        v = fb_v + self.ff(bundle=terms)

        # guard: an estimator still converging from zero must not produce
        # NaN pwm; the norm clamp bounds the result either way
        pwm = (v / jnp.maximum(est_v, 1e-3)).clamp_norm(1.0)

        self.fets(est_i.norm2())                  # fets own r_dson: amps^2 in
        self.pwm_prev(pwm)
        return pwm

    # the signature declares the spec, exactly as at a leaf
    return members(apply)


# --- command controllers: command -> current target ---

@node
def torque_command():
    """Torque command -> DQ current target (id = 0)."""
    def apply(command, motor):
        iq = command / motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)
    return Leaf(apply)


@node
def VelocityCommand(velocity_ctrl: Node) -> Node:
    """Velocity command -> torque (via the velocity PID member) ->
    DQ current target. Runs on ESTIMATED mechanicals: the observer's
    output, not the true state."""
    members = Composite(velocity_ctrl=velocity_ctrl)

    def apply(self, command, est_mechanical, motor):
        torque = self.velocity_ctrl(command - est_mechanical.velocity)
        iq = torque / motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)

    return members(apply)
