"""Mechanics, constraint schedules, and simulated rollout for an XPBD chain of 2D squares."""

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

from nodejax import Struct, batch, scan, split_aux
from examples.pbd import (
    AnchorConstraint,
    Broadcast,
    body,
    gauss_seidel,
    jacobi,
    red_black,
    xpbd_step,
)


GRAVITY = (0.0, -9.81)


def square_chain_bodies(
    n_squares: int,
    size: float,
    angle: float = 0.0,
    mass: float = 1.0,
) -> Struct:
    """Build a chain of 2D square bodies rotated by ``angle`` with the first square fixed."""
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    rotation = jnp.array([[cosine, -sine], [sine, cosine]])
    centers_x = (jnp.arange(n_squares) + 0.5) * size
    local_positions = jnp.stack((centers_x, jnp.zeros(n_squares)), axis=-1)
    positions = local_positions @ rotation.T
    angles = jnp.full(n_squares, angle)
    velocities = jnp.zeros((n_squares, 2))
    angular_velocities = jnp.zeros(n_squares)

    # Square 0 is pinned; subsequent squares are free
    inv_mass = jnp.concatenate((jnp.zeros(1), jnp.full(n_squares - 1, 1.0 / mass)))
    inertia = (mass * size ** 2) / 6.0
    inv_inertia = jnp.concatenate((jnp.zeros(1), jnp.full(n_squares - 1, 1.0 / inertia)))

    return body(
        position=positions,
        angle=angles,
        velocity=velocities,
        angular_velocity=angular_velocities,
        inverse_mass=inv_mass,
        inverse_inertia=inv_inertia,
    )


def square_chain_constraints(
    n_squares: int,
    size: float,
    compliance: float = 0.0,
) -> Struct:
    """Build consecutive hinge-pin constraints joining adjacent square edges."""
    n_constraints = n_squares - 1
    indices = jnp.stack((jnp.arange(n_constraints), jnp.arange(1, n_squares)), axis=-1)
    anchors = jnp.tile(
        jnp.array([[size / 2.0, 0.0], [-size / 2.0, 0.0]]),
        (n_constraints, 1, 1),
    )

    return Struct(
        index=indices,
        constraint=Struct(
            anchors=anchors,
            rest_length=jnp.zeros(n_constraints),
            compliance=jnp.full(n_constraints, compliance),
        ),
    )


def test_one_anchor_constraint_restores_offset_bodies() -> None:
    # Body 0 fixed at origin; Body 1 displaced to (3, 0).
    # Anchors at (+1, 0) and (-1, 0) should meet at (1, 0).
    initial = body(
        position=jnp.array([[0.0, 0.0], [3.0, 0.0]]),
        angle=jnp.array([0.0, 0.0]),
        velocity=jnp.zeros((2, 2)),
        angular_velocity=jnp.zeros(2),
        inverse_mass=jnp.array([0.0, 1.0]),
        inverse_inertia=jnp.array([0.0, 6.0]),
    )
    spec = Struct(
        anchors=jnp.array([[1.0, 0.0], [-1.0, 0.0]]),
        rest_length=0.0,
        compliance=0.0,
    )
    constraint = AnchorConstraint().bind(spec)

    after, aux = split_aux(constraint.apply(initial))

    assert jnp.allclose(after.position[0], jnp.array([0.0, 0.0]))
    assert jnp.allclose(after.position[1], jnp.array([2.0, 0.0]))
    assert jnp.allclose(after.angle, 0.0)
    assert jnp.allclose(aux.distance_error, 1.0)


@pytest.mark.parametrize(
    'schedule',
    (gauss_seidel, jacobi, red_black),
    ids=('gauss-seidel', 'jacobi', 'red-black'),
)
def test_chain_schedules_reduce_the_same_stretched_chain(
    schedule: Callable,
) -> None:
    n_squares = 4
    size = 0.2
    # Start with squares placed further apart than their size
    centers_x = (jnp.arange(n_squares) + 0.5) * (size * 1.5)
    initial = square_chain_bodies(n_squares, size).replace(
        position=jnp.stack((centers_x, jnp.zeros(n_squares)), axis=-1)
    )
    constraints = square_chain_constraints(n_squares, size)
    step = xpbd_step(
        schedule(constraints, AnchorConstraint()),
        Broadcast((0.0, 0.0)),
        n_bodies=n_squares,
        n_solver_passes=8,
        dt=0.02,
        velocity_damping=1.0,
    )

    final = split_aux(step.parameterize().bind(state=initial).apply()[1])[0]

    # Check anchor separation between first pair
    init_anchor_0 = initial.position[0] + jnp.array([size / 2.0, 0.0])
    init_anchor_1 = initial.position[1] + jnp.array([-size / 2.0, 0.0])
    init_dist = jnp.linalg.norm(init_anchor_1 - init_anchor_0)

    final_anchor_0 = final.position[0] + jnp.array([size / 2.0, 0.0])
    final_anchor_1 = final.position[1] + jnp.array([-size / 2.0, 0.0])
    final_dist = jnp.linalg.norm(final_anchor_1 - final_anchor_0)

    assert final_dist < init_dist
    assert jnp.allclose(final.position[0], initial.position[0])


def test_square_chain_swings_under_gravity() -> None:
    n_squares = 4
    size = 0.2
    n_steps = 60
    initial = square_chain_bodies(n_squares, size)
    constraints = square_chain_constraints(n_squares, size)
    step = xpbd_step(
        gauss_seidel(constraints, AnchorConstraint()),
        Broadcast(GRAVITY),
        n_bodies=n_squares,
        n_solver_passes=10,
        dt=0.016,
        velocity_damping=0.995,
    )
    program = scan(step, n=n_steps)
    sim = jax.jit(program.parameterize().bind(state=initial).apply)
    trajectory, diagnostics = split_aux(sim()[1])

    assert trajectory.position.shape == (n_steps, n_squares, 2)
    assert trajectory.angle.shape == (n_steps, n_squares)
    assert jnp.all(jnp.isfinite(trajectory.position))
    assert jnp.all(jnp.isfinite(trajectory.angle))
    # Fixed root does not move
    assert jnp.allclose(trajectory.position[:, 0], initial.position[0])
    # Tip falls downward under gravity
    assert trajectory.position[-1, -1, 1] < initial.position[-1, 1]


def test_composed_batched_rollout_keeps_anchor() -> None:
    n_squares = 4
    n_worlds = 2
    n_steps = 60
    size = 0.2
    angles = jnp.array([-0.3, 0.3])
    initial = jax.vmap(
        lambda angle: square_chain_bodies(n_squares, size, angle)
    )(angles)
    constraints = square_chain_constraints(n_squares, size)
    step = xpbd_step(
        gauss_seidel(constraints, AnchorConstraint()),
        Broadcast(GRAVITY),
        n_bodies=n_squares,
        n_solver_passes=8,
        dt=0.016,
        velocity_damping=0.995,
    )
    program = batch(scan(step, n=n_steps), n=n_worlds)
    sim = jax.jit(program.parameterize().bind(state=initial).apply)
    trajectory, diagnostics = split_aux(sim()[1])

    assert trajectory.position.shape == (n_worlds, n_steps, n_squares, 2)
    assert trajectory.angle.shape == (n_worlds, n_steps, n_squares)
    assert jnp.all(jnp.isfinite(trajectory.position))
    # Fixed root stays in place for both worlds
    assert jnp.allclose(trajectory.position[:, :, 0], initial.position[:, None, 0])


def plot_squares(result: Struct, filename: str = 'xpbd_squares.png') -> str:
    """Plot multi-world simulated frames and physical energy decay of swinging 2D rigid squares."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    size = float(result.size)
    half = size / 2.0
    corner_template = np.array([
        [-half, -half],
        [half, -half],
        [half, half],
        [-half, half],
        [-half, -half],
    ])

    positions = np.array(result.trajectory.position)
    angles = np.array(result.trajectory.angle)
    velocities = np.array(result.trajectory.velocity)
    angular_velocities = np.array(result.trajectory.angular_velocity)
    release_angles = np.array(result.angles)
    n_worlds, n_frames, n_squares = positions.shape[:3]

    figure = plt.figure(figsize=(14.0, 7.8))
    grid = figure.add_gridspec(2, n_worlds, height_ratios=[1.35, 1.0])

    frame_indices = np.linspace(0, n_frames - 1, 7, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(frame_indices)))

    # Top row: geometry for each world
    for world_idx in range(n_worlds):
        axis = figure.add_subplot(grid[0, world_idx])
        world_positions = positions[world_idx]
        world_angles = angles[world_idx]

        # Draw tip trajectory trail
        axis.plot(
            world_positions[:, -1, 0],
            world_positions[:, -1, 1],
            ':',
            color='0.6',
            linewidth=1.0,
            label='tip trail',
        )

        for frame_idx, color in zip(frame_indices, colors):
            frame_positions = world_positions[frame_idx]
            frame_angles = world_angles[frame_idx]
            for square_idx in range(n_squares):
                cosine = np.cos(frame_angles[square_idx])
                sine = np.sin(frame_angles[square_idx])
                rotation = np.array([[cosine, -sine], [sine, cosine]])
                corners = (rotation @ corner_template.T).T + frame_positions[square_idx]
                label = (
                    f'{frame_idx * float(result.dt):.2f} s'
                    if square_idx == 0
                    else None
                )
                axis.fill(corners[:, 0], corners[:, 1], facecolor=color, alpha=0.15)
                axis.plot(
                    corners[:, 0],
                    corners[:, 1],
                    '-',
                    color=color,
                    linewidth=1.2,
                    label=label,
                )
                # Pin marker at the right edge
                pin = frame_positions[square_idx] + rotation @ np.array([half, 0.0])
                axis.plot(pin[0], pin[1], 'o', color=color, markersize=3.0)

        # Pin marker on fixed root
        axis.scatter(*world_positions[0, 0], marker='s', color='black', s=45, zorder=5)
        axis.set_aspect('equal')
        axis.grid(alpha=0.2)
        axis.set_xlabel('x')
        if world_idx == 0:
            axis.set_ylabel('y')
        axis.set_title(f'released at {float(release_angles[world_idx]):+.2f} rad')
        if world_idx == n_worlds - 1:
            axis.legend(title='time', frameon=False, fontsize=7.5, loc='lower right')

    # Bottom row: Energy curves and constraint violation
    # Compute energy for world 1 (horizontal release)
    world_index = 1
    mass = 1.0
    inertia = (mass * size ** 2) / 6.0
    # Kinetic energy: 0.5 * m * v^2 + 0.5 * I * w^2 (summed over free bodies 1..N-1)
    linear_kinetic_energy = 0.5 * mass * np.sum(
        velocities[world_index, :, 1:] ** 2,
        axis=(-1, -2),
    )
    rotational_kinetic_energy = 0.5 * inertia * np.sum(
        angular_velocities[world_index, :, 1:] ** 2,
        axis=-1,
    )
    kinetic_energy = linear_kinetic_energy + rotational_kinetic_energy
    # Gravitational potential energy: m * g * y
    potential_energy = np.sum(
        mass * 9.81 * positions[world_index, :, 1:, 1],
        axis=-1,
    )
    total_energy = kinetic_energy + potential_energy
    time = np.arange(n_frames) * float(result.dt)

    energy_axis = figure.add_subplot(grid[1, :2])
    energy_axis.plot(time, total_energy, 'k-', linewidth=1.6, label='total energy (E)')
    energy_axis.plot(
        time,
        kinetic_energy,
        'b--',
        linewidth=1.2,
        label='kinetic (T)',
    )
    energy_axis.plot(
        time,
        potential_energy,
        'r--',
        linewidth=1.2,
        label='potential (V)',
    )
    energy_axis.set_xlabel('time (s)')
    energy_axis.set_ylabel('energy (J)')
    energy_axis.set_title('energy trace (horizontal release): smooth dissipation via damping')
    energy_axis.grid(alpha=0.2)
    energy_axis.legend(frameon=False, fontsize=8)

    # Bottom-right: constraint violation across time
    error_axis = figure.add_subplot(grid[1, 2])
    # Extract max distance error from aux diagnostics
    error = np.array(result.aux.solve.constraint.distance_error)
    # error shape: (n_worlds, n_frames, n_passes, n_constraints)
    max_error_per_frame = np.max(np.abs(error[world_index, :, -1, :]), axis=-1)
    error_axis.semilogy(
        time,
        np.maximum(max_error_per_frame, 1e-8),
        color='teal',
        linewidth=1.3,
    )
    error_axis.set_xlabel('time (s)')
    error_axis.set_ylabel('max anchor error (m)')
    error_axis.set_title('joint error (last solver pass)')
    error_axis.grid(alpha=0.2)

    figure.suptitle(
        'XPBD 2D rigid squares: linear/angular anchors, repeated time, batched worlds',
        fontweight='bold',
    )
    figure.tight_layout()
    output = Path(__file__).parents[1] / 'plots' / filename
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def main() -> None:
    n_squares = 4
    n_worlds = 3
    n_steps = 150
    size = 0.2
    dt = 0.016

    angles = jnp.array([-0.4, 0.0, 0.4])
    initial = jax.vmap(
        lambda angle: square_chain_bodies(n_squares, size, angle)
    )(angles)
    constraints = square_chain_constraints(n_squares, size)

    step = xpbd_step(
        gauss_seidel(constraints, AnchorConstraint()),
        Broadcast(GRAVITY),
        n_bodies=n_squares,
        n_solver_passes=12,
        dt=dt,
        velocity_damping=0.995,
    )
    program = batch(scan(step, n=n_steps), n=n_worlds)
    sim = jax.jit(program.parameterize().bind(state=initial).apply)
    trajectory, aux = split_aux(sim()[1])

    result = Struct(
        program=program,
        initial=initial,
        trajectory=trajectory,
        aux=aux,
        dt=dt,
        size=size,
        angles=angles,
    )
    print(program.tree_view())
    print(f'plot: {plot_squares(result)}')


if __name__ == '__main__':
    main()
