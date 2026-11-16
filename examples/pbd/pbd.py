"""Defines particle records, distance/bending/floor constraints, and `pbd_step`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    cyclic,
    node,
    repeat,
)
from nodejax.struct import register_struct_subtype
from examples.pbd.core import PBDStep, safe_norm


@register_struct_subtype
class Particle(Struct):
    """Stores particle `position`, `velocity`, and `inverse_mass` arrays with PBD coordinate properties."""

    @property
    def coords(self) -> Struct:
        return Struct(position=self.position)

    @property
    def inv_inertia(self) -> Struct:
        return Struct(position=self.inverse_mass[:, None])


def particle(
    position: jax.Array,
    velocity: jax.Array,
    inverse_mass: jax.Array,
) -> Particle:
    """Constructs a `Particle` record whose zero `inverse_mass` entries anchor particles in place."""
    return Particle(
        position=position,
        velocity=velocity,
        inverse_mass=inverse_mass,
    )


@node
def FreeMotion(dt: float) -> PNode:
    """Advances one `Particle` unconstrained over `dt` under external `force`."""
    def apply(particle: Particle, force: jax.Array) -> Particle:
        velocity = particle.velocity + dt * particle.inverse_mass * force
        return particle.replace(
            position=particle.position + dt * velocity,
            velocity=velocity,
        )

    return Leaf(apply)


@node
def VelocityUpdate(
    dt: float,
    damping: float,
) -> PNode:
    """Recovers damped particle velocity from `(projected.position - previous.position) / dt`."""
    def apply(previous: Particle, projected: Particle) -> Particle:
        velocity = damping * (projected.position - previous.position) / dt
        return projected.replace(velocity=velocity)

    return Leaf(apply)


@node
def ParticleDistance() -> Node:
    """Evaluates scalar distance error `||x_1 - x_0|| - rest_length` for a particle pair."""
    def param(rest_length: float, compliance: float = 0.0) -> Struct:
        return Struct(rest_length=rest_length, compliance=compliance)

    def apply(param, coords: Struct) -> jax.Array:
        return safe_norm(coords.position[1] - coords.position[0]) - param.rest_length

    return Leaf(apply, param=param)


@node
def ParticleBend() -> Node:
    """Evaluates signed 2D turning-angle error `atan2(cross, dot) - rest_angle` across a particle triple."""
    def param(rest_angle: float = 0.0, compliance: float = 0.0) -> Struct:
        return Struct(rest_angle=rest_angle, compliance=compliance)

    def apply(param, coords: Struct) -> jax.Array:
        first = coords.position[1] - coords.position[0]
        second = coords.position[2] - coords.position[1]
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        return jnp.arctan2(cross, dot) - param.rest_angle

    return Leaf(apply, param=param)


@node
def FloorConstraint(height: float) -> PNode:
    """Clamps the vertical coordinate `position[1]` of one `Particle` at or above `height`."""
    def apply(particle: Particle) -> Particle:
        position = particle.position.at[1].set(jnp.maximum(particle.position[1], height))
        return particle.replace(position=position)

    return Leaf(apply)


def pbd_step(
    constraints: BaseNode,
    *,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float = 1.0,
) -> Node | PNode:
    """Builds one cyclic particle PBD timestep carrying the `Particle` collection as state."""
    predict = batch(FreeMotion(dt))
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(VelocityUpdate(dt, velocity_damping))
    return cyclic(PBDStep(predict, solver, finalize))
