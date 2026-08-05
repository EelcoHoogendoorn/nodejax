"""PMSM motor model.

One param constructor (motor_params), one shared METHODS dict
(MOTOR_METHODS: the PMSM model equations, DQ arithmetic), three carriers:
plant_def (full dynamics), emotor_def (electrical only), and
motor_model_def — a STATELESS node that exists to be the internal-model
FIELD of a controller: params + methods as a bound Node, state held
elsewhere. Being non-cyclic its state slot is (), so composites can hold
it without acquiring state.

The cogging harmonic amplitude is the motor param `cogging`, so
clean-plant tests can zero it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import ambient, node_def

from nodejax.examples.actuator.dq import DQ


def motor_params(resistance=0.24, inductance_d=2e-4, inductance_q=3e-4,
                 kt=1.2, pole_pairs=16.0, slots=36.0, inertia=1e-1,
                 friction=1e-3, hysteresis=0.0, torque_static=0.0,
                 cogging=0.8, dedent_offset=0.0) -> Struct:
    """One physical motor object; defaults describe a small direct-drive outrunner."""
    fields = dict(resistance=resistance, inductance_d=inductance_d,
                  inductance_q=inductance_q, kt=kt, pole_pairs=pole_pairs,
                  slots=slots, inertia=inertia, friction=friction,
                  hysteresis=hysteresis, torque_static=torque_static,
                  cogging=cogging, dedent_offset=dedent_offset)
    return Struct(**{k: jnp.asarray(v, dtype=jnp.float32) for k, v in fields.items()})


# --- phase-neutral conversions (dq params carry a 2/3 factor) ---

def _R(m):
    return m.resistance * (2.0 / 3.0)


def _L(m):
    return DQ(m.inductance_d * (2.0 / 3.0), m.inductance_q * (2.0 / 3.0))


def _flux_linkage(m):
    return m.kt / m.pole_pairs / 1.5


# --- the model equations; module-level as
# implementation, attached to every motor-shaped node as METHODS ---

def voltage_terms(m, i: DQ, di_dt: DQ, velocity_mech) -> Struct:
    """The PMSM voltage equation, SPLIT: resistive drop, inductive
    transient, and speed voltage (back-EMF + reluctance coupling) — so a
    controller can weight its trust per term."""
    omega = velocity_mech * m.pole_pairs
    L = _L(m)
    return Struct(resistive=i * _R(m),
                  inductive=di_dt * L,
                  bemf=DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(m))) * omega)


def voltage_feedforward(m, i: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """V = R*I + L*dI/dt + omega * flux coupling (the fused sum)."""
    terms = voltage_terms(m, i, di_dt, velocity_mech)
    return terms.resistive + terms.inductive + terms.bemf


def current_feedforward(m, v: DQ, i: DQ, velocity_mech) -> DQ:
    """dI/dt = (V - R*I - omega * flux coupling) / L."""
    omega = velocity_mech * m.pole_pairs
    L = _L(m)
    v_omega = DQ(d=-(L.q * i.q), q=(L.d * i.d + _flux_linkage(m))) * omega
    return (v - i * _R(m) - v_omega) / L


def current_model(m, v: DQ, di_dt: DQ, velocity_mech) -> DQ:
    """I = (V - L*dI/dt - omega*flux) / R — the inverse voltage equation
    (q-axis-only back-EMF approximation), used by the model-based
    current estimator."""
    omega = velocity_mech * m.pole_pairs
    v_omega = DQ(d=jnp.zeros_like(omega), q=_flux_linkage(m) * omega)
    return (v - di_dt * _L(m) - v_omega) / _R(m)


def _saturation_factor(amps):
    saturation = 80.0
    softmaxout = lambda x, y, h: jnp.log(jnp.exp(x * h) + jnp.exp(y * h)) / h
    return softmaxout(1.0, (jnp.abs(amps) / saturation - 1.0) / 4.0 + 1.0, 20.0)


def torque(m, i: DQ):
    """Output torque from DQ current, with magnetic saturation applied."""
    L = _L(m)
    return 1.5 * m.pole_pairs * (
        _flux_linkage(m) * i.q + (L.d - L.q) * i.d * i.q
    ) / _saturation_factor(i.q)


def _dedent(m, angle):
    """Cogging torque harmonics (pole and slot orders)."""
    aa = angle + m.dedent_offset
    poles = m.pole_pairs * 2.0
    h = lambda f: jnp.sin(aa * f) * m.cogging
    return (h(poles * 0.5) + h(poles) + h(poles * 2.0)
            + h(m.slots) + h(m.slots * 2.0) + h(m.slots * 3.0))


MOTOR_METHODS = dict(voltage_feedforward=voltage_feedforward,
                     voltage_terms=voltage_terms,
                     current_feedforward=current_feedforward,
                     current_model=current_model,
                     torque=torque)


def motor_model_def():
    """The motor as a pure MODEL object: no state, no dynamics — a
    carrier for params + methods, to live as a field inside controllers
    and estimators. Its apply is the model's map,
    current -> torque."""
    def apply(self, input):
        return torque(self, input)
    return node_def(apply, param=motor_params, name='motor_model',
                    methods=MOTOR_METHODS)


# --- the physical plant as a cyclic node ---

@ambient
def plant_def(dt, substeps=4):
    """Voltage command (DQ) + load torque -> motor state; `self` is the
    motor. Electrical + mechanical Euler substeps, stiction, cogging,
    hysteresis. State carries the current as a DQ."""
    def init(position=0.0):
        return Struct(current=DQ(jnp.asarray(0.0), jnp.asarray(0.0)),
                      position=jnp.asarray(position), velocity=jnp.asarray(0.0))

    def _mechanical(m, s, load, h):
        tau = torque(m, s.current)
        friction = m.friction * s.velocity + m.hysteresis * jnp.sign(s.velocity)
        total = tau - friction - load + _dedent(m, s.position)
        velocity = s.velocity + (total / m.inertia) * h
        # stiction: stick to zero when below the static torque threshold
        is_dynamic = jnp.abs(velocity) * m.inertia > m.torque_static * h
        velocity = velocity * is_dynamic
        position = jnp.mod(s.position + velocity * h, 2.0 * jnp.pi)
        return s.replace(position=position, velocity=velocity)

    def apply(self, state, input):
        h = dt / substeps

        def substep(_, s):
            di_dt = current_feedforward(self, input.v, s.current, s.velocity)
            s = s.replace(current=s.current + di_dt * h)
            return _mechanical(self, s, input.load, h)

        new = jax.lax.fori_loop(0, substeps, substep, state)
        return new, new

    return node_def(apply, init=init, param=motor_params, name='plant',
                    methods=MOTOR_METHODS)


# --- split plant: electrical motor + external mechanics ---

@ambient
def emotor_def(dt, substeps=4):
    """Electrical-only motor: mechanics live outside the actuator,
    integrated at the environment level. Voltage in, torque and current
    out; `self` is the motor."""
    def init():
        return DQ(jnp.asarray(0.0), jnp.asarray(0.0))

    def apply(self, state, input):
        h = dt / substeps

        def substep(_, i):
            di_dt = current_feedforward(self, input.v, i, input.mechanical.velocity)
            return i + di_dt * h

        current = jax.lax.fori_loop(0, substeps, substep, state)
        return current, Struct(torque=torque(self, current), current=current)

    return node_def(apply, init=init, param=motor_params, name='emotor',
                    methods=MOTOR_METHODS)


@ambient
def mechanical_def(dt):
    """External mechanics: torque + load -> position/velocity."""
    def param(inertia, friction=0.0):
        return Struct(inertia=jnp.asarray(inertia), friction=jnp.asarray(friction))

    def init(param, position=0.0):
        return Struct(position=jnp.asarray(position), velocity=jnp.asarray(0.0))

    def apply(self, state, input):
        total = input.torque - self.friction * state.velocity - input.load
        velocity = state.velocity + (total / self.inertia) * dt
        position = jnp.mod(state.position + velocity * dt, 2.0 * jnp.pi)
        new = Struct(position=position, velocity=velocity)
        return new, new

    return node_def(apply, init=init, param=param, name='mechanical')
