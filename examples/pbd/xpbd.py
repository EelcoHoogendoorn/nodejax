"""Extended Position-Based Dynamics (XPBD) for 2D rigid bodies.

Provides the ``body`` record with linear and rotational degrees of freedom,
unconstrained rigid motion (``FreeRigidMotion``), velocity recovery
(``RigidVelocityUpdate``), the anchor-to-anchor constraint kernel (``AnchorConstraint``),
and the composed rigid timestep ``xpbd_step``.
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
)
from examples.pbd.core import PBDStep


def body(
    position: jax.Array,
    angle: jax.Array,
    velocity: jax.Array,
    angular_velocity: jax.Array,
    inverse_mass: jax.Array,
    inverse_inertia: jax.Array,
) -> Struct:
    """Build a 2D extended rigid body record.

    ``position`` and ``velocity`` are 2D center-of-mass coordinates. ``angle``
    and ``angular_velocity`` describe 2D rotation. ``inverse_mass`` and
    ``inverse_inertia`` are scalars (zero fixes the corresponding degree of freedom).

    A record, not a Node: it flows as data through every operation of a
    timestep and is the state of nothing beneath ``PBDStep``, which alone
    holds the collection as its state (see ``core``).
    """
    return Struct(
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
def AnchorConstraint() -> Node:
    """Enforce distance between local anchor points on a pair of 2D rigid bodies.

    Param is a record with ``anchors`` of shape ``(2, 2)``, ``rest_length``, and
    ``compliance``. Solves coupled linear and rotational corrections along the
    line of action.
    """
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

    def apply(param, pair):
        rotated_anchors = _rotate_2d(pair.angle, param.anchors)
        anchor_positions = pair.position + rotated_anchors
        anchor_offset = anchor_positions[1] - anchor_positions[0]
        distance = jnp.linalg.norm(anchor_offset)
        direction = anchor_offset / jnp.maximum(distance, 1e-8)
        error = distance - param.rest_length

        moment = (
            rotated_anchors[..., 0] * direction[1]
            - rotated_anchors[..., 1] * direction[0]
        )
        effective_inverse_mass = pair.inverse_mass + pair.inverse_inertia * (moment ** 2)
        total_effective_inverse_mass = jnp.sum(effective_inverse_mass) + param.compliance

        multiplier = error / jnp.maximum(total_effective_inverse_mass, 1e-8)
        signs = jnp.array([1.0, -1.0])

        position = pair.position + (signs * pair.inverse_mass)[:, None] * direction * multiplier
        angle = pair.angle + signs * pair.inverse_inertia * moment * multiplier

        return pair.replace(position=position, angle=angle), Aux(distance_error=error)

    return Leaf(apply, param=param)


def xpbd_step(
    constraints: Node | PNode,
    forcing: Node,
    *,
    n_bodies: int,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float,
) -> Node:
    """Assemble the rigid body world: one physical timestep, cyclic over the
    collection it steps, so a caller binds or initializes its state and
    scans it over time."""
    predict = batch(FreeRigidMotion(dt), n=n_bodies)
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(RigidVelocityUpdate(dt, velocity_damping), n=n_bodies)
    return cyclic(PBDStep(forcing, predict, solver, finalize))
