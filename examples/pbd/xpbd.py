"""Extended Position-Based Dynamics (XPBD) for 2D rigid bodies.

Provides the ``body`` record with linear and rotational degrees of freedom,
unconstrained rigid motion (``FreeRigidMotion``), velocity recovery
(``RigidVelocityUpdate``), the anchor-to-anchor constraint kernel
(``AnchorDistance``), and the composed rigid timestep ``xpbd_step``.
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
class Body(Struct):
    """A 2D rigid body record of poses, velocities, and inverse inertias."""

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
    """A 2D rigid body record of poses, velocities, and inverse inertias.

    ``position`` and ``velocity`` are center-of-mass vectors; ``angle`` and
    ``angular_velocity`` are scalars. Zero ``inverse_mass`` or ``inverse_inertia``
    fixes that degree of freedom. Leading batch axes describe collections.

    Individual substeps pass this record as call data; ``xpbd_step`` holds the
    collection as state across timesteps via ``cyclic``.
    """
    return Body(
        position=position,
        angle=angle,
        velocity=velocity,
        angular_velocity=angular_velocity,
        inverse_mass=inverse_mass,
        inverse_inertia=inverse_inertia,
    )


@node
def FreeRigidMotion(dt: float) -> Node:
    """Predict one unconstrained 2D rigid ``body`` record from external force."""
    def apply(body, force):
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
) -> Node:
    """Reconstruct linear and angular velocities from corrected rigid body poses."""
    def apply(previous, projected):
        velocity = damping * (projected.position - previous.position) / dt
        angular_velocity = damping * (projected.angle - previous.angle) / dt
        return projected.replace(
            velocity=velocity,
            angular_velocity=angular_velocity,
        )

    return Leaf(apply)


def _rotate_2d(angle: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotate 2D ``vector`` values by scalar ``angle`` radians."""
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
    """Compute distance error between local anchor points on a pair of 2D rigid bodies."""
    def param(
        anchors: jax.Array,
        rest_length: float = 0.0,
        compliance: float = 0.0,
    ):
        return Struct(
            anchors=anchors,
            rest_length=rest_length,
            compliance=compliance,
        )

    def apply(param, coords):
        rotated_anchors = _rotate_2d(coords.angle, param.anchors)
        delta_pos = coords.position[1] - coords.position[0]
        delta_anchor = rotated_anchors[1] - rotated_anchors[0]
        return safe_norm(delta_pos + delta_anchor) - param.rest_length

    return Leaf(apply, param=param)


def xpbd_step(
    constraints: Node | PNode,
    *,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float = 1.0,
) -> Node:
    """Assemble one 2D rigid body timestep, cyclic over the body collection."""
    predict = batch(FreeRigidMotion(dt))
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(RigidVelocityUpdate(dt, velocity_damping))
    return cyclic(PBDStep(predict, solver, finalize))
