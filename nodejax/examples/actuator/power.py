"""Battery and lumped thermal models.

The battery is read in two different ways each step, and the method
mechanism carries that: voltage(charge) is a METHOD (a pure read of the
sag curve at the current charge), while apply() is the step (discharge
by the power drawn). capacity=inf gives an always-charged supply.
"""

from __future__ import annotations

import jax.numpy as jnp

import jax

from nodejax.struct import Struct
from nodejax import Node, node, ambient, Leaf, derive

from nodejax.examples.actuator.utils import lerp


@node
def Battery(dt: float) -> Node:
    def param(voltage_max=48.0, voltage_min=0.0, capacity=jnp.inf):
        return Struct(voltage_max=voltage_max,
                      voltage_min=voltage_min,
                      capacity=capacity)

    def init(param):
        return 1.0   # charge, [0, 1]

    def apply(self, state, input):
        # input: power drawn (W); an empty battery stays empty (floor at 0)
        charge = jnp.maximum(state - input * dt / self.capacity, 0.0)
        return charge, charge

    def voltage(param, state):
        """The sag curve at the current charge — a pure read, no step.
        The `state` parameter name declares the role: reached through a
        composite's self, the live charge binds automatically."""
        return lerp(param.voltage_min, param.voltage_max, state)

    return Leaf(apply, init=init, param=param,
                    methods=dict(voltage=voltage))


@node
def Thermal(dt: float) -> Node:
    """Single-node lumped thermal model: dissipated power in, temperature
    out, exponential relaxation to ambient; one node per component."""
    def param(r_th, c_th, ambient=25.0):
        return Struct(r_th=r_th, c_th=c_th,
                      ambient=ambient)

    def init(param):
        return param.ambient

    def apply(self, state, input):
        t = state + (input - (state - self.ambient) / self.r_th) / self.c_th * dt
        return t, t

    return Leaf(apply, init=init, param=param)


@node
def DeratingThermal(dt: float) -> Node:
    """The thermal model plus the derating story, via derive(): a temperature
    limit and a sigmoid derating curve as a METHOD — whoever owns this
    node's state derates a current by it (read-then-step). The thermal
    dynamics are inherited untouched."""
    parent = Thermal(dt)

    def param(r_th, c_th, limit, hardness=4.0, ambient=25.0):
        return parent.parameterize(
            r_th=r_th, c_th=c_th, ambient=ambient).param.replace(
            limit=limit, hardness=hardness)

    def derate(param, state, current):
        # state is the node's temperature; the name declares the role
        return current * jax.nn.sigmoid(
            (param.limit - state) / param.limit * param.hardness)

    return derive(parent, param=param, methods=dict(derate=derate),
                  name='derating_thermal')


@node
def FET(dt: float) -> Node:
    """The power stage: derived once more — it owns its conduction losses
    (r_dson), so apply takes the SQUARED CURRENT magnitude and converts to
    watts itself; super is the explicit parent apply call."""
    parent = DeratingThermal(dt)

    def param(r_th, c_th, limit=80.0, r_dson=0.02, hardness=4.0, ambient=25.0):
        return parent.parameterize(
            r_th=r_th, c_th=c_th, limit=limit,
            hardness=hardness, ambient=ambient).param.replace(
            r_dson=r_dson)

    def apply(self, state, input):
        return parent.apply(self.param, state, input * self.r_dson)

    return derive(parent, param=param, apply=apply, name='fets')
