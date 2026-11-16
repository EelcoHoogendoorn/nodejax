"""Models PMSM `Electrical` dynamics, `Mechanical` load integration, and closed-loop `BenchMotor` composition."""

from __future__ import annotations

import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, node, Leaf, Composite

from examples.actuator.dq import DQ


# Phase-to-neutral conversions (DQ parameters carry a 2/3 factor).

def _R(param: Struct):
    return param.resistance * (2.0 / 3.0)


def _L(param: Struct) -> DQ:
    return DQ(param.inductance_d * (2.0 / 3.0), param.inductance_q * (2.0 / 3.0))


def _flux_linkage(param: Struct):
    return param.kt / param.pole_pairs / 1.5


def voltage_terms(param: Struct, i: DQ, di_dt: DQ, velocity_mech) -> Struct:
    """Decomposes the PMSM voltage equation into `Struct(resistive, inductive, bemf)` terms."""
    omega = velocity_mech * param.pole_pairs
    L = _L(param)
    return Struct(resistive=i * _R(param),
                  inductive=di_dt * L,
                  bemf=DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(param))) * omega)


def voltage_feedforward(param: Struct, i: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """Sums the resistive, inductive, and back-EMF voltage terms (`R * I + L * dI/dt + omega * flux`)."""
    terms = voltage_terms(param, i, di_dt, velocity_mech)
    return terms.resistive + terms.inductive + terms.bemf


def current_feedforward(param: Struct, v: DQ, i: DQ, velocity_mech) -> DQ:
    """Computes the instantaneous current derivative `(V - R * I - omega * flux) / L`."""
    omega = velocity_mech * param.pole_pairs
    L = _L(param)
    omega_v = DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(param))) * omega
    return (v - i * _R(param) - omega_v) / L


def current_step(param: Struct, i: DQ, v: DQ, velocity_mech, h: float) -> DQ:
    """Advances `DQ` current by `h` seconds at voltage `v` via a 2x2 backward-Euler solve."""
    omega = velocity_mech * param.pole_pairs
    R = _R(param)
    L = _L(param)
    # Solve (I - h * A) i_new = i + h * b for the coupled d/q RL circuit at electrical speed omega.
    diagonal_d = 1.0 + h * R / L.d
    diagonal_q = 1.0 + h * R / L.q
    coupling_d = -h * omega * L.q / L.d
    coupling_q = h * omega * L.d / L.q
    forced_d = i.d + h * v.d / L.d
    forced_q = i.q + h * (v.q - omega * _flux_linkage(param)) / L.q
    determinant = diagonal_d * diagonal_q - coupling_d * coupling_q
    return DQ(
        d=(diagonal_q * forced_d - coupling_d * forced_q) / determinant,
        q=(diagonal_d * forced_q - coupling_q * forced_d) / determinant,
    )


def current_model(param: Struct, v: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """Estimates `DQ` current `(V - L * dI/dt - omega * flux) / R` under a `q`-axis back-EMF approximation."""
    omega = velocity_mech * param.pole_pairs
    omega_v = DQ(d=jnp.zeros_like(omega), q=_flux_linkage(param) * omega)
    return (v - di_dt * _L(param) - omega_v) / _R(param)


def _saturation_factor(amps):
    saturation = 80.0
    softmaxout = lambda x, y, h: jnp.log(jnp.exp(x * h) + jnp.exp(y * h)) / h
    return softmaxout(1.0, (jnp.abs(amps) / saturation - 1.0) / 4.0 + 1.0, 20.0)


def torque(param: Struct, i: DQ):
    """Computes electromagnetic shaft torque from `DQ` current including reluctance and magnetic saturation."""
    L = _L(param)
    return 1.5 * param.pole_pairs * (
        _flux_linkage(param) * i.q + (L.d - L.q) * i.d * i.q
    ) / _saturation_factor(i.q)


def _dedent(param: Struct, angle):
    """Evaluates pole-order and slot-order cogging torque harmonics at rotor `angle`."""
    shifted = angle + param.dedent_offset
    poles = param.pole_pairs * 2.0
    harmonic = lambda freq: jnp.sin(shifted * freq) * param.cogging
    return (harmonic(poles * 0.5) + harmonic(poles) + harmonic(poles * 2.0)
            + harmonic(param.slots) + harmonic(param.slots * 2.0) + harmonic(param.slots * 3.0))


MOTOR_METHODS = dict(voltage_feedforward=voltage_feedforward,
                     voltage_terms=voltage_terms,
                     current_feedforward=current_feedforward,
                     current_step=current_step,
                     current_model=current_model,
                     torque=torque)


@node
def Electrical(dt: float) -> Node:
    """Steps `DQ` stator currents implicitly over `dt` and emits `Struct(torque, current)` given `(mechanical, voltage)`.

    Attaches `MOTOR_METHODS` so the same `Electrical` definition serves both as the physical plant and as the controller's internal motor model. Includes optional pole/slot cogging harmonics and Coulomb hysteresis drag while leaving inertia and stiction to `Mechanical`.
    """
    def param(resistance, inductance_d, inductance_q, kt, pole_pairs, slots,
              hysteresis=0.0, cogging=0.0, dedent_offset=0.0) -> Struct:
        return Struct(resistance=resistance, inductance_d=inductance_d,
                      inductance_q=inductance_q, kt=kt, pole_pairs=pole_pairs,
                      slots=slots, hysteresis=hysteresis,
                      cogging=cogging, dedent_offset=dedent_offset)

    def init() -> DQ:
        return DQ(0.0, 0.0)

    def apply(param, state, mechanical: Struct, voltage: DQ) -> tuple[DQ, Struct]:
        current = current_step(param, state, voltage, mechanical.velocity, dt)
        tq = (torque(param, current) + _dedent(param, mechanical.position)
              - param.hysteresis * jnp.sign(mechanical.velocity))
        return current, Struct(torque=tq, current=current)

    return Leaf(apply, param=param, init=init,
                methods=MOTOR_METHODS)


@node
def Mechanical(dt: float) -> Node:
    """Integrates rotor `Struct(position, velocity)` under shaft `torque`, viscous `friction`, `torque_static` stiction, and external `load`."""
    def param(inertia, friction=0.0, torque_static=0.0) -> Struct:
        return Struct(inertia=inertia, friction=friction,
                      torque_static=torque_static)

    def init(position=0.0) -> Struct:
        return Struct(position=position, velocity=0.0)

    def apply(param, state, torque, load) -> tuple[Struct, Struct]:
        total = torque - param.friction * state.velocity - load
        velocity = state.velocity + (total / param.inertia) * dt
        is_dynamic = jnp.abs(velocity) * param.inertia > param.torque_static * dt
        velocity = velocity * is_dynamic
        position = jnp.mod(state.position + velocity * dt, 2.0 * jnp.pi)
        new = Struct(position=position, velocity=velocity)
        return new, new

    return Leaf(apply, init=init, param=param)


@node
def BenchMotor(dt: float) -> Node:
    """Couples `Electrical(dt)` and `Mechanical(dt)` into an open-loop bench motor from `(voltage, load)` to `Struct(position, velocity)`."""
    members = Composite(electrical=Electrical(dt), mechanical=Mechanical(dt))

    def apply(self, voltage: DQ, load) -> Struct:
        out = self.electrical(mechanical=self.mechanical.state, voltage=voltage)
        return self.mechanical(torque=out.torque, load=load)

    return members(apply)
