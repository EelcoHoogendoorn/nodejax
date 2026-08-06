"""Current controller: target current + measurements in, PWM out.

Composes a model-based estimator, one DQ PID, the internal motor model
(a param-only member: the controller's model of the plant), the power
stage's own thermal (fets), the per-term feedforward weighting (ff) and
a norm-clamp current limiter (limit), and two one-tick memories:
previous pwm feeding the estimator's model, previous target feeding the
reference-derivative inductive feedforward. Output is a PWM command —
voltage normalized by the ESTIMATED bus voltage, norm-clamped to 1.
The fets heat on their own conduction losses (read-then-step) and
derate the target; derating for components this controller does NOT
own (the motor windings) happens upstream, where those thermals live.
"""

from __future__ import annotations

import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import ambient, node_def, composite

from nodejax.examples.actuator.dq import DQ
from nodejax.examples.actuator.blocks import clamp_norm_def, delay_def
from nodejax.examples.actuator.motor import current_model, voltage_terms


@ambient
def foc_current_model(dt):
    """Predict the current the previously commanded modulated voltage
    implies, with di/dt taken filtered-vs-previous — the physics as
    plain functions on plain param data."""
    def model_fn(filtered, previous, model):
        di_dt = (filtered - previous) / dt
        return current_model(model.motor, model.v_mod, di_dt, model.velocity)
    return model_fn


def ff_def():
    """Per-term feedforward: the model's voltage terms weighted by the
    trust placed in each (resistive/bemf/inductive gains), summed. A
    leaf node — the terms its input, the gains its params."""
    def param(r, bemf, l):
        return Struct(r=jnp.asarray(r), bemf=jnp.asarray(bemf), l=jnp.asarray(l))

    def apply(self, input):
        return (input.resistive * self.r + input.bemf * self.bemf
                + input.inductive * self.l)

    return node_def(apply, param=param, name='ff')


@ambient
def current_controller_def(dt, motor, estimator, controller, fets):
    """Target current + measurements in, PWM out.

    input fields, declared by the apply signature: current (true DQ
    current), velocity (estimated mechanical velocity), bus (estimated
    bus voltage), target (DQ current target). The target arrives PRE-DERATED for anything this controller does
    not own (the motor's thermal rollback happens where the motor
    thermal lives — at the stack). The factory argument list is the
    member list; ff (per-term feedforward weighting) and limit (a
    norm-clamp on the target current) are leaf-node members like the
    rest."""
    members = dict(motor=motor, estimator=estimator, controller=controller,
                   fets=fets, ff=ff_def()(r=1.0, bemf=1.0, l=0.0),
                   limit=clamp_norm_def()(limit=100.0),
                   pwm_prev=delay_def(DQ()), tgt_prev=delay_def(DQ()))

    def apply(self, current, velocity, bus, target):
        target = self.limit(target)
        # the power stage derates by its own temperature (read-then-step)
        target = self.fets.derate(target)

        # model-based estimation: the model sees the voltage we actually
        # commanded last step (previous pwm x estimated bus)
        i_est = self.estimator(Struct(
            value=current,
            model=Struct(motor=self.param.motor,
                         v_mod=self.state.pwm_prev * bus,
                         velocity=velocity)))

        v_fb = self.controller(target - i_est)
        # feedforward with PER-TERM trust weights: resistive and
        # speed-voltage terms at the target operating point, and the
        # inductive term driven by the REFERENCE derivative — an
        # error-driven di/dt is a hidden P-gain of L/dt on measurement
        # noise (measured: tracking flat, jitter x4)
        di_ref = (target - self.state.tgt_prev) / dt
        terms = self.motor.voltage_terms(target, di_ref, velocity)
        v = v_fb + self.ff(terms)

        # guard: an estimator still converging from zero must not produce
        # NaN pwm; the norm clamp bounds the result either way
        pwm = (v / jnp.maximum(bus, 1e-3)).clamp_norm(1.0)

        self.fets(i_est.norm2())                  # fets own r_dson: amps^2 in
        self.pwm_prev(pwm)
        self.tgt_prev(target)
        return pwm

    # the signature declares the spec, exactly as at a leaf
    return composite(apply, members=members, name='current_controller')


# --- command controllers: command -> current target ---

def torque_command_def():
    """Torque command -> DQ current target (id = 0)."""
    def apply(input):
        iq = input.command / input.motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)
    return node_def(apply, name='torque_command')


def velocity_command_def(velocity_ctrl):
    """Velocity command -> torque (via the velocity PID member) ->
    DQ current target."""
    members = dict(velocity_ctrl=velocity_ctrl)

    def apply(self, input):
        torque = self.velocity_ctrl(input.command - input.mechanical.velocity)
        iq = torque / input.motor.kt
        return DQ(d=jnp.zeros_like(iq), q=iq)

    return composite(apply, members=members, name='velocity_command')
