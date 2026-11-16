"""Models battery discharge curves and lumped thermal derating Nodes (`Battery`, `Thermal`, `DeratingThermal`, `FET`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, node, Leaf, derive

from examples.actuator.utils import lerp


@node
def Battery(dt: float) -> Node:
    """Discharges normalized state of charge in `[0, 1]` by drawn power and exposes open-circuit `voltage` as a method."""
    def param(voltage_max, voltage_min, capacity) -> Struct:
        return Struct(voltage_max=voltage_max,
                      voltage_min=voltage_min,
                      capacity=capacity)

    def init(param) -> float:
        return 1.0   # charge, [0, 1]

    def apply(param, state, input):
        # input: power drawn (W); floor state of charge at 0.
        charge = jnp.maximum(state - input * dt / param.capacity, 0.0)
        return charge, charge

    def voltage(param, state):
        """Interpolates bus voltage between `voltage_min` and `voltage_max` at the current state of charge."""
        return lerp(param.voltage_min, param.voltage_max, state)

    return Leaf(apply, init=init, param=param,
                methods=dict(voltage=voltage))


@node
def Thermal(dt: float) -> Node:
    """Integrates a single-node lumped temperature relaxing toward `param.ambient` with thermal resistance `r_th` and capacitance `c_th`."""
    def param(r_th, c_th, ambient=25.0) -> Struct:
        return Struct(r_th=r_th, c_th=c_th,
                      ambient=ambient)

    def init(param):
        return param.ambient

    def dissipation(param, input):
        return input

    def apply(node, param, state, input):
        power = node.dissipation(param, input)
        cooling = (state - param.ambient) / param.r_th
        temperature = state + (power - cooling) / param.c_th * dt
        return temperature, temperature

    return Leaf(
        apply, init=init, param=param,
        methods=dict(dissipation=dissipation),
    )


@node
def DeratingThermal(dt: float) -> Node:
    """Extends `Thermal(dt)` via `derive` with a `limit` temperature and a smooth sigmoid `derate(current)` method."""
    parent = Thermal(dt)

    def param(limit, hardness=4.0) -> Struct:
        return Struct(limit=limit, hardness=hardness)

    def derate(param, state, current):
        return current * jax.nn.sigmoid(
            (param.limit - state) / param.limit * param.hardness)

    return derive(parent, param=param, methods=dict(derate=derate),
                  name='derating_thermal')


@node
def FET(dt: float) -> Node:
    """Extends `DeratingThermal(dt)` via `derive` to convert squared current (`I^2`) into dissipated power through `param.r_dson`."""
    parent = DeratingThermal(dt)

    def param(r_dson) -> Struct:
        return Struct(r_dson=r_dson)

    def dissipation(param, input):
        return input * param.r_dson

    return derive(
        parent, param=param,
        methods=dict(dissipation=dissipation), name='fets',
    )
