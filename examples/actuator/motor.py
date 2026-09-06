"""PMSM motor model.

Electrical is THE motor: params, the model equations as methods,
and the electrical dynamics. The same def serves every role — the plant
in the assembled stack, half of BenchMotor (its composition with
the mechanism), and the controller's internal model, where it is held
as a member and consulted through its methods and params. The model
equations are module-level functions on the param bundle; the node
attaches them as methods.

The cogging harmonic amplitude is the motor param `cogging`, so
clean-plant tests can zero it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, node, ambient, Leaf, Composite

from examples.actuator.dq import DQ


# --- phase-neutral conversions (dq params carry a 2/3 factor) ---

def _R(param):
    return param.resistance * (2.0 / 3.0)


def _L(param):
    return DQ(param.inductance_d * (2.0 / 3.0), param.inductance_q * (2.0 / 3.0))


def _flux_linkage(param):
    return param.kt / param.pole_pairs / 1.5


# --- the model equations; module-level as
# implementation, attached to every motor-shaped node as METHODS ---

def voltage_terms(param, i: DQ, di_dt: DQ, velocity_mech) -> Struct:
    """The PMSM voltage equation, SPLIT: resistive drop, inductive
    transient, and speed voltage (back-EMF + reluctance coupling) — so a
    controller can weight its trust per term."""
    omega = velocity_mech * param.pole_pairs
    L = _L(param)
    return Struct(resistive=i * _R(param),
                  inductive=di_dt * L,
                  bemf=DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(param))) * omega)


def voltage_feedforward(param, i: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """V = R*I + L*dI/dt + omega * flux coupling (the fused sum)."""
    terms = voltage_terms(param, i, di_dt, velocity_mech)
    return terms.resistive + terms.inductive + terms.bemf


def current_feedforward(param, v: DQ, i: DQ, velocity_mech) -> DQ:
    """dI/dt = (V - R*I - omega * flux coupling) / L."""
    omega = velocity_mech * param.pole_pairs
    L = _L(param)
    omega_v = DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(param))) * omega
    return (v - i * _R(param) - omega_v) / L


def current_step(param, i: DQ, v: DQ, velocity_mech, h: float) -> DQ:
    """The current after ``h`` seconds at voltage ``v``, by one implicit
    Euler step of the voltage equation. The equation is linear in the
    currents at a given speed, so the step is a two by two solve, and it
    is stable at any ``h``: stability first, accuracy second, the standard
    the actuator module holds its dynamics to."""
    omega = velocity_mech * param.pole_pairs
    R = _R(param)
    L = _L(param)
    # (I - h A) i_new = i + h b, with A the circuit's own matrix and b the
    # forcing from the voltage and the permanent magnet's speed voltage.
    diagonal_d = 1.0 + h * R / L.d
    diagonal_q = 1.0 + h * R / L.q
    coupling_d = -h * omega * L.q / L.d   # upper right of I - hA
    coupling_q = h * omega * L.d / L.q    # lower left of I - hA
    forced_d = i.d + h * v.d / L.d
    forced_q = i.q + h * (v.q - omega * _flux_linkage(param)) / L.q
    determinant = diagonal_d * diagonal_q - coupling_d * coupling_q
    return DQ(
        d=(diagonal_q * forced_d - coupling_d * forced_q) / determinant,
        q=(diagonal_d * forced_q - coupling_q * forced_d) / determinant,
    )


def current_model(param, v: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """I = (V - L*dI/dt - omega*flux) / R — the inverse voltage equation
    (q-axis-only back-EMF approximation), used by the model-based
    current estimator."""
    omega = velocity_mech * param.pole_pairs
    omega_v = DQ(d=jnp.zeros_like(omega), q=_flux_linkage(param) * omega)
    return (v - di_dt * _L(param) - omega_v) / _R(param)


def _saturation_factor(amps):
    saturation = 80.0
    softmaxout = lambda x, y, h: jnp.log(jnp.exp(x * h) + jnp.exp(y * h)) / h
    return softmaxout(1.0, (jnp.abs(amps) / saturation - 1.0) / 4.0 + 1.0, 20.0)


def torque(param, i: DQ):
    """Output torque from DQ current, with magnetic saturation applied."""
    L = _L(param)
    return 1.5 * param.pole_pairs * (
        _flux_linkage(param) * i.q + (L.d - L.q) * i.d * i.q
    ) / _saturation_factor(i.q)


def _dedent(param, angle):
    """Cogging torque harmonics (pole and slot orders)."""
    aa = angle + param.dedent_offset
    poles = param.pole_pairs * 2.0
    h = lambda f: jnp.sin(aa * f) * param.cogging
    return (h(poles * 0.5) + h(poles) + h(poles * 2.0)
            + h(param.slots) + h(param.slots * 2.0) + h(param.slots * 3.0))


MOTOR_METHODS = dict(voltage_feedforward=voltage_feedforward,
                     voltage_terms=voltage_terms,
                     current_feedforward=current_feedforward,
                     current_step=current_step,
                     current_model=current_model,
                     torque=torque)


# --- the electrical motor: mechanics live outside ---

@node
def Electrical(dt: float) -> Node:
    """The electrical motor: mechanics live outside (the mechanism's
    inertia and loads are integrated by Mechanical, at the bench or
    the environment), so position and velocity arrive as INPUT and
    torque goes out. The torque is everything the MOTOR produces:
    electromagnetic, cogging (position harmonics), and iron hysteresis
    drag; what the mechanism does with it (inertia, bearing friction,
    stiction) is the mechanism's. Voltage in, torque and current out.
    Electrical and geometric constants come from the motor configuration;
    optional loss effects default to zero. The currents advance by one
    implicit step per tick (see ``current_step``), so any ``dt`` is
    stable and the motor never needs substeps of its own."""
    def param(resistance, inductance_d, inductance_q, kt, pole_pairs, slots,
              hysteresis=0.0, cogging=0.0, dedent_offset=0.0):
        return Struct(resistance=resistance, inductance_d=inductance_d,
                      inductance_q=inductance_q, kt=kt, pole_pairs=pole_pairs,
                      slots=slots, hysteresis=hysteresis,
                      cogging=cogging, dedent_offset=dedent_offset)

    def init():
        return DQ(0.0, 0.0)

    def apply(param, state, mechanical, voltage: DQ):
        current = current_step(param, state, voltage, mechanical.velocity, dt)
        tq = (torque(param, current) + _dedent(param, mechanical.position)
              - param.hysteresis * jnp.sign(mechanical.velocity))
        return current, Struct(torque=tq, current=current)

    return Leaf(apply, param=param, init=init,
                    methods=MOTOR_METHODS)


@node
def Mechanical(dt: float) -> Node:
    """The mechanism: torque + load -> position/velocity. Owns what a
    mechanism owns: inertia, bearing friction, and stiction (stick to
    zero below the static torque threshold)."""
    def param(inertia, friction=0.0, torque_static=0.0):
        return Struct(inertia=inertia, friction=friction,
                      torque_static=torque_static)

    def init(position=0.0):
        return Struct(position=position, velocity=0.0)

    def apply(param, state, torque, load):
        total = torque - param.friction * state.velocity - load
        velocity = state.velocity + (total / param.inertia) * dt
        is_dynamic = jnp.abs(velocity) * param.inertia > param.torque_static * dt
        velocity = velocity * is_dynamic
        position = jnp.mod(state.position + velocity * dt, 2.0 * jnp.pi)
        new = Struct(position=position, velocity=velocity)
        return new, new

    return Leaf(apply, init=init, param=param)


# --- the motor on a bench: composition, the same parts the system uses ---

@node
def BenchMotor(dt: float) -> Node:
    """The whole motor on a bench, composed from the SAME parts the
    assembled system uses: the electrical motor plus the mechanism,
    in a feedback loop (previous mechanical state feeds the motor,
    the motor's torque steps the mechanism). Voltage + load in,
    mechanical state out."""
    members = Composite(electrical=Electrical(dt), mechanical=Mechanical(dt))

    def apply(self, voltage, load):
        out = self.electrical(mechanical=self.mechanical.state, voltage=voltage)
        return self.mechanical(torque=out.torque, load=load)

    return members(apply)
