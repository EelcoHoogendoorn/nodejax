"""Defines 2D rigid-body records, anchor-distance constraints, and `xpbd_step`."""

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
class Body(Struct):
    """Stores 2D rigid-body poses, linear and angular velocities, and inverse inertias."""

    @property
    def coords(self) -> Struct:
        return Struct(position=self.position, angle=self.angle)

    @property
    def inv_inertia(self) -> Struct:
        return Struct(position=self.inverse_mass[:, None], angle=self.inverse_inertia)


def body(
    position: jax.Array,
    angle: jax.Array,
    velocity: jax.Array,
    angular_velocity: jax.Array,
    inverse_mass: jax.Array,
    inverse_inertia: jax.Array,
) -> Body:
    """Constructs a 2D rigid `Body` record whose zero `inverse_mass` or `inverse_inertia` entries fix degrees of freedom."""
    return Body(
        position=position,
        angle=angle,
        velocity=velocity,
        angular_velocity=angular_velocity,
        inverse_mass=inverse_mass,
        inverse_inertia=inverse_inertia,
    )


@node
def FreeRigidMotion(dt: float) -> PNode:
    """Advances one 2D rigid `Body` unconstrained over `dt` under external linear `force`."""
    def apply(body: Body, force: jax.Array) -> Body:
        velocity = body.velocity + dt * body.inverse_mass * force
        return body.replace(
            position=body.position + dt * velocity,
            angle=body.angle + dt * body.angular_velocity,
            velocity=velocity,
        )

    return Leaf(apply)


@node
def RigidVelocityUpdate(
    dt: float,
    damping: float,
) -> PNode:
    """Recovers damped linear and angular velocities from projected 2D rigid-body pose displacements."""
    def apply(previous: Body, projected: Body) -> Body:
        velocity = damping * (projected.position - previous.position) / dt
        angular_velocity = damping * (projected.angle - previous.angle) / dt
        return projected.replace(
            velocity=velocity,
            angular_velocity=angular_velocity,
        )

    return Leaf(apply)


def _rotate_2d(angle: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotates 2D `vector` coordinates counter-clockwise by `angle` radians."""
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.stack(
        (
            cosine * vector[..., 0] - sine * vector[..., 1],
            sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        axis=-1,
    )


@node
def AnchorDistance() -> Node:
    """Evaluates scalar distance error between body-local `anchors` across a pair of 2D rigid bodies."""
    def param(
        anchors: jax.Array,
        rest_length: float = 0.0,
        compliance: float = 0.0,
    ) -> Struct:
        return Struct(
            anchors=anchors,
            rest_length=rest_length,
            compliance=compliance,
        )

    def apply(param, coords: Struct) -> jax.Array:
        rotated_anchors = _rotate_2d(coords.angle, param.anchors)
        delta_pos = coords.position[1] - coords.position[0]
        delta_anchor = rotated_anchors[1] - rotated_anchors[0]
        return safe_norm(delta_pos + delta_anchor) - param.rest_length

    return Leaf(apply, param=param)


def xpbd_step(
    constraints: BaseNode,
    *,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float = 1.0,
) -> Node | PNode:
    """Builds one cyclic 2D rigid-body XPBD timestep carrying the `Body` collection as state."""
    predict = batch(FreeRigidMotion(dt))
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(RigidVelocityUpdate(dt, velocity_damping))
    return cyclic(PBDStep(predict, solver, finalize))
