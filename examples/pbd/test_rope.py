"""Mechanics, composition checks, and simulated rollout for the PBD rope."""

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

from nodejax import Struct, batch, scan, split_aux, tree_broadcast_axis
from examples.pbd import (
    Broadcast,
    DistanceConstraint,
    gauss_seidel,
    jacobi,
    particle,
    pbd_step,
    red_black,
)


GRAVITY = (0.0, -9.81)


def rope_constraints(n_points: int, rest_length: float) -> Struct:
    """The distance constraints of a chain: consecutive particles joined at one rest length."""
    indices = jnp.stack((jnp.arange(n_points - 1), jnp.arange(1, n_points)), axis=-1)
    return Struct(
        index=indices,
        constraint=jnp.full(n_points - 1, rest_length),
    )


def initial_rope(
    n_points: int,
    rest_length: float,
    anchor_height: float,
    angle: float = 0.0,
) -> Struct:
    """A particle record forming a straight, left-fixed rope at rest,
    ``angle`` radians above horizontal."""
    reach = rest_length * jnp.arange(n_points)
    position = jnp.stack((
        reach * jnp.cos(angle),
        anchor_height + reach * jnp.sin(angle),
    ), axis=-1)
    inverse_mass = jnp.concatenate((jnp.zeros((1,)), jnp.ones((n_points - 1,))))
    return particle(
        position=position,
        velocity=jnp.zeros_like(position),
        inverse_mass=inverse_mass,
    )


def test_one_distance_constraint_restores_a_pinned_pair() -> None:
    before = particle(
        position=jnp.array(((0.0, 0.0), (2.0, 0.0))),
        velocity=jnp.zeros((2, 2)),
        inverse_mass=jnp.array((0.0, 1.0)),
    )
    projection = DistanceConstraint().bind(1.0)

    after, aux = split_aux(projection.apply(before))

    assert jnp.allclose(after.position[0], before.position[0])
    assert jnp.allclose(after.position[1], jnp.array((1.0, 0.0)))
    assert jnp.allclose(aux.length_error, 1.0)


@pytest.mark.parametrize(
    'schedule',
    (gauss_seidel, jacobi, red_black),
    ids=('gauss-seidel', 'jacobi', 'red-black'),
)
def test_constraint_schedules_reduce_the_same_stretched_rope(
    schedule: Callable,
) -> None:
    n_points = 6
    initial_rest_length = 0.3
    target_rest_length = 0.2
    initial = initial_rope(n_points, initial_rest_length, anchor_height=1.0)
    step = pbd_step(
        schedule(rope_constraints(n_points, target_rest_length), DistanceConstraint()),
        Broadcast((0.0, 0.0)),
        n_points=n_points,
        n_solver_passes=8,
        dt=0.02,
        floor_height=-1.0,
        velocity_damping=1.0,
    )
    final = split_aux(step.parameterize().bind(state=initial).apply()[1])[0]
    initial_segment = initial.position[1:] - initial.position[:-1]
    final_segment = final.position[1:] - final.position[:-1]
    initial_squared_error = jnp.sum(
        (jnp.linalg.norm(initial_segment, axis=-1) - target_rest_length) ** 2
    )
    final_squared_error = jnp.sum(
        (jnp.linalg.norm(final_segment, axis=-1) - target_rest_length) ** 2
    )

    assert final_squared_error < initial_squared_error
    assert jnp.allclose(final.position[0], initial.position[0])


def test_force_free_rope_at_rest_remains_at_rest() -> None:
    n_points = 5
    n_steps = 8
    initial = initial_rope(n_points, rest_length=0.2, anchor_height=1.0)
    step = pbd_step(
        gauss_seidel(rope_constraints(n_points, 0.2), DistanceConstraint()),
        Broadcast((0.0, 0.0)),
        n_points=n_points,
        n_solver_passes=3,
        dt=0.02,
        floor_height=-1.0,
        velocity_damping=1.0,
    )
    program = scan(step, n=n_steps)
    sim = jax.jit(program.parameterize().bind(state=initial).apply)
    trajectory, diagnostics = split_aux(sim()[1])

    assert jnp.allclose(trajectory.position, initial.position)
    assert jnp.allclose(trajectory.velocity, 0.0)
    assert diagnostics.solve.constraints.constraint.length_error.shape == (
        n_steps,
        3,
        n_points - 1,
    )


def test_composed_rollout_keeps_the_anchor_and_floor() -> None:
    n_points = 8
    n_worlds = 2
    n_steps = 80
    floor_height = 0.0
    initial = initial_rope(n_points, rest_length=0.15, anchor_height=0.8)
    initial = tree_broadcast_axis(initial, n_worlds, axis=0)
    step = pbd_step(
        gauss_seidel(rope_constraints(n_points, 0.15), DistanceConstraint()),
        Broadcast(GRAVITY),
        n_points=n_points,
        n_solver_passes=8,
        dt=0.02,
        floor_height=floor_height,
        velocity_damping=0.995,
    )
    program = batch(scan(step, n=n_steps), n=n_worlds)
    sim = jax.jit(program.parameterize().bind(state=initial).apply)
    trajectory, diagnostics = split_aux(sim()[1])

    assert trajectory.position.shape == (n_worlds, n_steps, n_points, 2)
    assert diagnostics.solve.constraints.constraint.length_error.shape == (
        n_worlds,
        n_steps,
        8,
        n_points - 1,
    )
    assert jnp.all(jnp.isfinite(trajectory.position))
    assert jnp.allclose(trajectory.position[:, :, 0], initial.position[:, None, 0])
    assert jnp.min(trajectory.position[..., 1]) >= floor_height - 1e-6
    segment = trajectory.position[..., 1:, :] - trajectory.position[..., :-1, :]
    length_error = jnp.abs(jnp.linalg.norm(segment, axis=-1) - 0.15)
    assert jnp.max(length_error) < 0.02


def plot_rope(result: Struct, filename: str = 'pbd_rope.png') -> str:
    """Save several real simulated frames for every world."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    initial_position = np.array(result.initial.position)
    simulated_position = np.array(result.trajectory.position)
    position = np.concatenate((initial_position[:, None], simulated_position), axis=1)
    n_worlds, n_frames = position.shape[:2]
    frame_indices = np.linspace(0, n_frames - 1, 7, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(frame_indices)))
    figure, axes = plt.subplots(
        1,
        n_worlds,
        figsize=(13.2, 4.2),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for world_index, axis in enumerate(axes):
        for frame_index, color in zip(frame_indices, colors):
            frame = position[world_index, frame_index]
            axis.plot(
                frame[:, 0],
                frame[:, 1],
                'o-',
                color=color,
                markersize=2.8,
                linewidth=1.2,
                label=f'{frame_index * float(result.dt):.1f} s',
            )
        axis.axhline(float(result.floor_height), color='0.2', linewidth=1.2)
        axis.scatter(*initial_position[world_index, 0], marker='s', color='black')
        axis.set_title(f'released at {float(result.angles[world_index]):+.2f} rad')
        axis.set_xlabel('x')
        axis.grid(alpha=0.2)
        axis.set_aspect('equal', adjustable='box')

    axes[0].set_ylabel('y')
    axes[-1].legend(title='simulated time', frameon=False, fontsize=8)
    figure.suptitle(
        'PBD rope: local constraints, repeated projection, scanned time, batched worlds',
        fontweight='bold',
    )
    figure.tight_layout()
    output = Path(__file__).parents[1] / 'plots' / filename
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def main() -> None:
    n_points = 14
    n_worlds = 3
    n_steps = 180
    rest_length = 0.12
    floor_height = 0.0
    anchor_height = 1.45
    dt = 0.016

    constraints = rope_constraints(n_points, rest_length)
    step = pbd_step(
        gauss_seidel(constraints, DistanceConstraint()),
        Broadcast(GRAVITY),
        n_points=n_points,
        n_solver_passes=8,
        dt=dt,
        floor_height=floor_height,
        velocity_damping=0.995,
    )
    program = batch(scan(step, n=n_steps), n=n_worlds)
    angles = jnp.linspace(-0.5, 0.5, n_worlds)
    initial = jax.vmap(
        lambda angle: initial_rope(n_points, rest_length, anchor_height, angle))(angles)
    trajectory, aux = split_aux(jax.jit(program.parameterize().bind(state=initial).apply)()[1])
    result = Struct(
        program=program,
        initial=initial,
        trajectory=trajectory,
        aux=aux,
        dt=dt,
        floor_height=floor_height,
        angles=angles,
    )
    print(program.tree_view())
    print(f'plot: {plot_rope(result)}')


if __name__ == '__main__':
    main()
