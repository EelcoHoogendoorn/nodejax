"""Run Gaussian MPPI with a learned Pendulum terminal critic and save its plot."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    BaseNode,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    carried,
    control,
    drop_aux,
    node,
    serial,
    tree_broadcast_axis,
    tree_len,
    tree_take,
)
from nodejax import nn
from examples.rl.control import policy_trajectory
from examples.rl.mppi import mppi_controller, mppi_learner, mppi_training
from examples.rl.pendulum import (
    PHASE_VELOCITY_LIMIT,
    Pendulum,
    PendulumCritic,
    downward_starts,
    overlay_phase_trajectories,
    phase_grid,
    phase_portrait,
    phase_starts,
    phase_surface,
)


DISCOUNT = 0.97
TRACE = 0.95
HIDDEN = 64
COMMAND_LIMIT = 3.0
NOISE_SCALE = 2.0
NOISE_CORRELATION = 0.9
TEMPERATURE = 0.3

N_ITERATIONS = 400
N_TRAINING_WORLDS = 16
N_TRAINING_REFINEMENTS = 2
N_TRAINING_CANDIDATES = 32
N_TRAINING_STEPS_PER_PLAN = 10
N_STEPS_PER_ITERATION = 50
N_CRITIC_UPDATES = 4
EMA_CRITIC_DECAY = 0.95
CRITIC_RATE = 0.001

N_EVALUATION_CANDIDATES = 64
N_EVALUATION_REFINEMENTS = 4
N_EVALUATION_STEPS = 300


@node
def PendulumCommandRange() -> Node:
    """Keep raw planner commands in a useful range before actuation."""
    return Leaf(lambda input: jnp.clip(input, -COMMAND_LIMIT, COMMAND_LIMIT))


@node
def PlanningStarts(
    n_iterations: int,
    n_worlds: int,
    n_steps_per_iteration: int,
) -> Node:
    """Mix hard reference starts with random states for every training chunk.

    The reference order alternates the negative and positive representations
    of the downward angle, then moves inward by equal offsets. Truncating the
    references for a small world batch therefore still covers both sides of
    the angle wrap. Remaining worlds start uniformly across the plotted phase
    region. Every start is repeated across time because ``ControlledStep``
    consumes it only when initializing a fresh sampled rollout.
    """
    def apply(rng):
        reference_order = jnp.asarray((0, 3, 1, 4, 2, 5))
        n_references = min(tree_len(downward_starts()), n_worlds)
        n_random_starts = n_worlds - n_references
        samples = jax.random.uniform(
            rng.next(),
            (2, n_iterations, n_random_starts),
        )
        random_starts = Struct(
            angle=2.0 * jnp.pi * samples[0] - jnp.pi,
            velocity=PHASE_VELOCITY_LIMIT * (2.0 * samples[1] - 1.0),
        )
        reference_starts = tree_broadcast_axis(
            tree_take(downward_starts(), reference_order[:n_references]),
            n_iterations,
            axis=0,
        )
        starts = jax.tree.map(
            lambda reference, sampled: jnp.concatenate(
                (reference, sampled), axis=1),
            reference_starts,
            random_starts,
        )
        return Struct(
            initial_state=tree_broadcast_axis(
                starts,
                n_steps_per_iteration,
                axis=2,
            ),
            disturbance=jnp.zeros(
                (n_iterations, n_worlds, n_steps_per_iteration)),
        )

    return Leaf(apply)


def pendulum_critic() -> Node:
    """Build an analytic phase-cost prior plus a learned residual."""
    residual = (
        nn.Linear(HIDDEN)
        >> nn.tanh
        >> nn.Linear(HIDDEN)
        >> nn.tanh
        >> nn.Projection()
    )
    return PendulumCritic(residual).with_input(
        Pendulum().initialize().state,
    )


def pendulum_training_program(
    n_iterations: int,
    critic: Node,
    plant: BaseNode,
    *,
    n_worlds: int = N_TRAINING_WORLDS,
    n_refinements: int = N_TRAINING_REFINEMENTS,
    n_candidates: int = N_TRAINING_CANDIDATES,
    n_steps_per_plan: int = N_TRAINING_STEPS_PER_PLAN,
    n_steps_per_iteration: int = N_STEPS_PER_ITERATION,
    n_critic_updates: int = N_CRITIC_UPDATES,
) -> Node:
    """Assemble the complete Pendulum MPPI critic-training tree."""
    controller = pendulum_controller(
        critic,
        plant,
        n_candidates=n_candidates,
        n_refinements=n_refinements,
        n_steps_per_plan=n_steps_per_plan,
    )
    learner = mppi_learner(
        controller,
        critic,
        plant,
        critic_optimizer=optax.adam(CRITIC_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        discount=DISCOUNT,
        trace=TRACE,
        n_worlds=n_worlds,
        n_steps_per_iteration=n_steps_per_iteration,
        n_critic_updates=n_critic_updates,
    )
    data = PlanningStarts(n_iterations, n_worlds, n_steps_per_iteration)
    return serial(data=data, training=carried(learner))


def pendulum_controller(
    critic: BaseNode,
    plant: BaseNode,
    *,
    n_candidates: int,
    n_refinements: int,
    n_steps_per_plan: int,
) -> Node:
    """The receding MPPI controller with the pendulum's command range and noise."""
    return mppi_controller(
        critic,
        plant,
        clean=PendulumCommandRange(),
        noise_scale=NOISE_SCALE,
        noise_correlation=NOISE_CORRELATION,
        temperature=TEMPERATURE,
        discount=DISCOUNT,
        n_candidates=n_candidates,
        n_refinements=n_refinements,
        n_steps_per_plan=n_steps_per_plan,
    )


def receding_trajectory(
    controller: PNode,
    starts: Struct,
    plant: BaseNode,
    *,
    steps: int,
    key: jax.Array,
) -> Struct:
    """Run real closed-loop trajectories with carried plan state."""
    return policy_trajectory(controller, plant, starts, steps, key)


def save_figure(figure, filename: str) -> str:
    import os

    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    return output


def plot_learned_critic(trajectory: Struct, critic: BaseNode) -> str:
    """Plot real receding rollouts beside the learned terminal value."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (phase_axis, value_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        sharey=True,
    )
    grid = phase_grid()
    phase_portrait(phase_axis, grid)
    overlay_phase_trajectories(phase_axis, trajectory.state)
    phase_axis.set_title('receding closed-loop rollouts', color='0.35', fontsize=10)

    flat = jax.tree.map(lambda value: value.reshape(-1), grid)
    values = drop_aux(batch(critic, n=tree_len(flat)).apply(flat))
    phase_surface(
        value_axis,
        grid,
        values.reshape(grid.angle.shape),
        label='learned terminal cost',
    )
    value_axis.set_title('learned terminal value', color='0.35', fontsize=10)
    figure.suptitle(
        'Receding Gaussian MPPI with a fitted terminal critic',
        y=0.98,
        fontweight='bold',
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output = save_figure(figure, 'pendulum_mppi_critic_phase_space.png')
    plt.close(figure)
    return output


def evaluate_receding(trajectory: Struct) -> Struct:
    """Summarize the reference starts without changing the plot sample."""
    references = tree_len(downward_starts())
    tail = jax.tree.map(
        lambda value: value[:references, -50:],
        trajectory.state,
    )
    upright = (jnp.abs(tail.angle) < 0.2) & (jnp.abs(tail.velocity) < 0.5)
    result = Struct(
        cost=jnp.mean(jnp.sum(trajectory.cost[:references], axis=1)),
        final_angle=jnp.max(jnp.abs(
            trajectory.final_state.angle[:references])),
        final_velocity=jnp.max(jnp.abs(
            trajectory.final_state.velocity[:references])),
        upright_fraction=jnp.mean(upright),
    )
    print(
        f'mean episode cost {result.cost:.2f}, '
        f'worst final angle {result.final_angle:.3f}, '
        f'worst final velocity {result.final_velocity:.3f}, '
        f'last-50 upright {100.0 * result.upright_fraction:.1f}%'
    )
    return result


def critic_swing_up() -> None:
    """Train a terminal critic, evaluate it in MPPI, and save one plot."""
    plant = Pendulum()
    critic = pendulum_critic()
    trained = mppi_training(
        pendulum_training_program(N_ITERATIONS, critic, plant),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    learned_critic = critic.bind(trained.learner.ema_critic.state)
    starts = phase_starts(
        jax.random.PRNGKey(29),
        velocity_limit=3.0,
    )
    learned_controller = pendulum_controller(
        learned_critic,
        plant,
        n_candidates=N_EVALUATION_CANDIDATES,
        n_refinements=N_EVALUATION_REFINEMENTS,
        n_steps_per_plan=N_TRAINING_STEPS_PER_PLAN,
    ).parameterize()
    learned = receding_trajectory(
        learned_controller,
        starts,
        plant,
        steps=N_EVALUATION_STEPS,
        key=jax.random.PRNGKey(4),
    )
    evaluate_receding(learned)
    print(f'phase plot: {plot_learned_critic(learned, learned_critic)}')


if __name__ == '__main__':
    critic_swing_up()
