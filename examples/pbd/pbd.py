"""Particle Position-Based Dynamics (PBD) entity definitions, constraints, and step.

Provides the ``particle`` record, unconstrained free motion, velocity recovery,
local particle constraints (``DistanceConstraint``, ``FloorConstraint``), and the
composed particle timestep ``pbd_step``.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    cyclic,
    node,
    repeat,
    serial,
)
from examples.pbd.core import PBDStep


def particle(
    position: jax.Array,
    velocity: jax.Array,
    inverse_mass: jax.Array,
) -> Struct:
    """Build a particle record used throughout the PBD solve.

    ``position`` and ``velocity`` end in a coordinate axis. ``inverse_mass``
    is scalar for one particle, and zero fixes that particle in place. Leading
    axes describe collections without changing the record's fields.

    A record, not a Node: it flows as data through every operation of a
    timestep and is the state of nothing beneath ``PBDStep``, which alone
    holds the collection as its state (see ``core``).
    """
    return Struct(
        position=position,
        velocity=velocity,
        inverse_mass=inverse_mass,
    )


@node
def FreeMotion(dt: float) -> Node:
    """Predict one unconstrained ``particle`` record from its force."""
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
def DistanceConstraint() -> Node:
    """Enforce rest length on a pair of particles; length error rides on aux."""
    def param(rest_length):
        return rest_length

    def apply(param, pair):
        offset = pair.position[1] - pair.position[0]
        distance = jnp.linalg.norm(offset)
        direction = offset / jnp.maximum(distance, 1e-8)
        length_error = distance - param
        total_inverse_mass = pair.inverse_mass[0] + pair.inverse_mass[1]
        correction = length_error * direction / jnp.maximum(total_inverse_mass, 1e-8)
        position = pair.position + jnp.stack((
            pair.inverse_mass[0] * correction,
            -pair.inverse_mass[1] * correction,
        ))
        return pair.replace(position=position), Aux(length_error=length_error)

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
    forcing: Node,
    *,
    n_points: int,
    n_solver_passes: int,
    dt: float,
    floor_height: float,
    velocity_damping: float,
) -> Node:
    """Assemble the particle world: one physical timestep, cyclic over the
    collection it steps, so a caller binds or initializes its state and
    scans it over time."""
    constraint_pass = serial(
        constraints=constraints,
        floor=batch(FloorConstraint(floor_height), n=n_points),
    )
    solver = repeat(constraint_pass, n=n_solver_passes)
    predict = batch(FreeMotion(dt), n=n_points)
    finalize = batch(VelocityUpdate(dt, velocity_damping), n=n_points)
    return cyclic(PBDStep(forcing, predict, solver, finalize))
