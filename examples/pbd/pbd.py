"""Particle Position-Based Dynamics (PBD) entity definitions, constraints, and step.

Provides the ``particle`` record, unconstrained free motion, velocity recovery,
local particle constraints (``ParticleDistance``, ``ParticleBend``,
``FloorConstraint``), and the composed particle timestep ``pbd_step``.
"""

import jax
import jax.numpy as jnp

from nodejax import (
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
    """A particle record of positions, velocities, and inverse masses."""

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
    """A particle record of positions, velocities, and inverse masses.

    ``position`` and ``velocity`` end in a coordinate axis. Zero ``inverse_mass``
    fixes that particle in place. Leading batch axes describe collections.

    Individual substeps pass this record as call data; ``pbd_step`` holds the
    collection as state across timesteps via ``cyclic``.
    """
    return Particle(
        position=position,
        velocity=velocity,
        inverse_mass=inverse_mass,
    )


@node
def FreeMotion(dt: float) -> Node:
    """Predict one unconstrained ``particle`` record from external force."""
    def apply(particle, force):
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
) -> Node:
    """Reconstruct one ``particle`` record from its corrected displacement."""
    def apply(previous, projected):
        velocity = damping * (projected.position - previous.position) / dt
        return projected.replace(velocity=velocity)

    return Leaf(apply)


@node
def ParticleDistance() -> Node:
    """Compute scalar distance error between a pair of particles."""
    def param(rest_length: float, compliance: float = 0.0):
        return Struct(rest_length=rest_length, compliance=compliance)

    def apply(param, coords):
        return safe_norm(coords.position[1] - coords.position[0]) - param.rest_length

    return Leaf(apply, param=param)


@node
def ParticleBend() -> Node:
    """The signed bending angle at the middle of a triple of 2D particles,
    less ``rest_angle``: the turn from the first segment to the second,
    zero when straight, with a clean gradient there."""
    def param(rest_angle: float = 0.0, compliance: float = 0.0):
        return Struct(rest_angle=rest_angle, compliance=compliance)

    def apply(param, coords):
        first = coords.position[1] - coords.position[0]
        second = coords.position[2] - coords.position[1]
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        return jnp.arctan2(cross, dot) - param.rest_angle

    return Leaf(apply, param=param)


@node
def FloorConstraint(height: float) -> Node:
    """Keep one ``particle`` record on or above a horizontal floor."""
    def apply(particle):
        position = particle.position.at[1].set(jnp.maximum(particle.position[1], height))
        return particle.replace(position=position)

    return Leaf(apply)


def pbd_step(
    constraints: Node | PNode,
    *,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float = 1.0,
) -> Node:
    """Assemble one particle timestep, cyclic over the particle collection."""
    predict = batch(FreeMotion(dt))
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(VelocityUpdate(dt, velocity_damping))
    return cyclic(PBDStep(predict, solver, finalize))
