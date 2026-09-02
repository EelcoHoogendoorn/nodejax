"""Analytic energy shaping for the shared weak pendulum.

This controller uses the mechanics that a learned policy must discover. The
swing phase regulates total energy toward the upright orbit. Once the pendulum
enters a small neighborhood of upright, a local proportional-derivative law
takes over. Persistent direction gives the exact hanging equilibrium a
deterministic kick, and a wider release band prevents repeated handoffs.

There are no learned parameters. The controller is still a real cyclic Node,
the plant is the same stateful Node used by SHAC and PPO, and ``scanned`` plus
``batch`` provide fresh state and independent rollouts.
"""

import jax
import jax.numpy as jnp

from nodejax import Leaf, Node, PNode, Struct, batch, node, scanned, tree_broadcast_axis, tree_last
from examples.rl.control import ControlledStep
from examples.rl.pendulum import (
    Pendulum,
    downward_starts,
    overlay_phase_trajectories,
    phase_grid,
    phase_portrait,
    phase_starts,
)


STEPS = 300


@node
def PendulumEnergyController(
    energy_gain: float = 8.0,
    energy_target: float = 0.05,
    kick: float = 0.05,
    position_gain: float = 4.0,
    velocity_gain: float = 2.0,
    capture_angle: float = 0.20,
    capture_velocity: float = 0.65,
    release_angle: float = 0.32,
    release_velocity: float = 1.0,
) -> Node:
    """Energy pumping with a hysteretic local balance mode."""

    def init(input):
        return Struct(
            balancing=jnp.zeros_like(input.angle, dtype=jnp.bool_),
            direction=jnp.where(input.angle < 0.0, 1.0, -1.0),
        )

    def apply(state, input):
        energy = 0.5 * input.velocity**2 + jnp.cos(input.angle) - 1.0
        swing_command = -energy_gain * (energy - energy_target) * (
            input.velocity + kick * state.direction
        )
        balance_command = (
            -position_gain * input.angle
            - velocity_gain * input.velocity
        )
        captured = (
            (jnp.abs(input.angle) < capture_angle)
            & (jnp.abs(input.velocity) < capture_velocity)
        )
        released = (
            (jnp.abs(input.angle) > release_angle)
            | (jnp.abs(input.velocity) > release_velocity)
        )
        balancing = jnp.where(state.balancing, ~released, captured)
        next_state = state.replace(balancing=balancing)
        command = jnp.where(balancing, balance_command, swing_command)
        return next_state, command

    return Leaf(apply, init=init)


def energy_shaping_program(worlds: int) -> PNode:
    """Build fresh parallel rollouts of the analytic controller."""
    world = ControlledStep(PendulumEnergyController(), Pendulum())
    return batch(scanned(world), n=worlds)


def energy_trajectory(
    initial_state: Struct,
    steps: int = STEPS,
) -> Struct:
    """Run deterministic closed-loop trajectories from supplied states."""
    worlds = initial_state.angle.shape[0]
    input = Struct(
        disturbance=jnp.zeros((worlds, steps)),
        initial_state=tree_broadcast_axis(initial_state, steps, axis=1),
    )
    return jax.jit(energy_shaping_program(worlds).apply)(bundle=input)


def plot_phase_space(trajectory: Struct) -> str:
    """Render real closed-loop rollouts from sampled phase states."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    phase_portrait(axis, phase_grid())
    overlay_phase_trajectories(axis, trajectory.state)
    axis.set_title(
        'pendulum energy shaping: real rollouts from sampled starts',
    )
    axis.legend(loc='upper right', frameon=False)
    figure.tight_layout()

    output = os.path.join(
        os.path.dirname(__file__),
        'plots',
        'pendulum_energy_shaping_phase_space.png',
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def test_controller_kicks_and_holds_the_balance_handoff() -> None:
    starts = Struct(
        angle=jnp.asarray((-jnp.pi, jnp.pi)),
        velocity=jnp.zeros(2),
    )
    controller = batch(PendulumEnergyController(), n=2).initialize(
        input=starts,
    )
    controller, command = controller.apply(starts)
    assert command[0] > 0.0
    assert command[1] < 0.0

    controller = PendulumEnergyController().initialize(
        input=Struct(angle=jnp.asarray(jnp.pi), velocity=jnp.asarray(0.0)),
    )
    controller, command = controller.apply(
        Struct(angle=jnp.asarray(0.1), velocity=jnp.asarray(0.1)),
    )
    assert controller.state.balancing
    assert command < 0.0
    controller, command = controller.apply(
        Struct(angle=jnp.asarray(0.25), velocity=jnp.asarray(0.7)),
    )
    assert controller.state.balancing
    controller, command = controller.apply(
        Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(0.7)),
    )
    assert not controller.state.balancing


def test_energy_shaping_swings_up_from_every_downward_start() -> None:
    trajectory = energy_trajectory(downward_starts())
    replay = energy_trajectory(downward_starts())

    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        trajectory,
        replay,
    ))
    assert jnp.max(jnp.abs(trajectory.next_state.angle[:, -50:])) < 1e-3
    assert jnp.max(jnp.abs(trajectory.next_state.velocity[:, -50:])) < 1e-3


if __name__ == '__main__':
    starts = phase_starts(jax.random.PRNGKey(29))
    trajectory = energy_trajectory(starts)
    final_state = tree_last(trajectory.next_state)
    print(energy_shaping_program(starts.angle.shape[0]).describe())
    print(
        'final error: '
        f'{jnp.max(jnp.abs(final_state.angle)):.6f} rad, '
        f'{jnp.max(jnp.abs(final_state.velocity)):.6f} rad/s',
    )
    print(f'phase portrait: {plot_phase_space(trajectory)}')
