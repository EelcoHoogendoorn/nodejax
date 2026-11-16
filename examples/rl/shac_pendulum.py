"""Trains a recurrent pendulum policy ensemble with SHAC and plots closed-loop phase trajectories."""

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
    drop_aux,
    ensemble,
    node,
    reduce,
    scan,
    serial,
    tree_len,
)
from nodejax import nn
from examples.rl import control
from examples.rl.pendulum import (
    MAX_TORQUE,
    VELOCITY_SCALE,
    Pendulum,
    PendulumCritic,
    PendulumMean,
    downward_starts,
    overlay_phase_trajectories,
    overlay_trajectories,
    phase_grid,
    phase_portrait,
    phase_starts,
    phase_surface,
)
from examples.rl.shac import shac_iteration, shac_training


DISCOUNT = 0.97
TRACE = 0.95
HIDDEN = 64
MEMORY = 64
N_POLICY_MEMBERS = 3
N_CRITIC_MEMBERS = 3
N_STEPS_PER_CHUNK = 20
N_WORLDS = 64
N_CHUNKS_PER_EPISODE = 15
N_EPISODES = 40
N_CRITIC_UPDATES = 4
EMA_CRITIC_DECAY = 0.995
ACTOR_RATE = 0.001
CRITIC_RATE = 0.001
DISTURBANCE_SCALE = 0.03
N_EVALUATION_STEPS = 300


@node
def ScalarMLP(hidden: int) -> Node:
    """Maps a feature vector to a scalar value through two `tanh` hidden layers."""
    return (
        nn.Linear(hidden)
        >> nn.tanh
        >> nn.Linear(hidden)
        >> nn.tanh
        >> nn.Projection()
    )


@node
def PendulumTrainingData(
    *,
    n_episodes: int,
    n_chunks_per_episode: int,
    n_steps_per_chunk: int,
    n_worlds: int,
    disturbance_scale: float,
) -> PNode:
    """Samples independent `(episode, world)` pendulum starts and `(episode, chunk, world, time)` torque disturbances."""
    def apply(rng):
        disturbance = disturbance_scale * jax.random.normal(
            rng.next(),
            (n_episodes, n_chunks_per_episode, n_worlds, n_steps_per_chunk),
        )
        plant_starts = Struct(
            angle=jax.random.uniform(
                rng.next(),
                (n_episodes, n_worlds),
                minval=-jnp.pi,
                maxval=jnp.pi,
            ),
            velocity=jax.random.uniform(
                rng.next(),
                (n_episodes, n_worlds),
                minval=-VELOCITY_SCALE,
                maxval=VELOCITY_SCALE,
            ),
        )
        # Nested scans slice every input field; only episode priming reads initial_plant_state.
        initial_plant_state = jax.tree.map(
            lambda value: jnp.broadcast_to(
                value[:, None, :, None],
                (n_episodes, n_chunks_per_episode, n_worlds, n_steps_per_chunk)
                + value.shape[2:],
            ),
            plant_starts,
        )
        return Struct(
            disturbance=disturbance,
            initial_plant_state=initial_plant_state,
        )

    return Leaf(apply)


def pendulum_policy(memory: BaseNode) -> Node:
    """Builds a `PendulumMean` policy wrapping `memory`."""
    return PendulumMean(memory=memory, hidden=HIDDEN)


def policy_committee(policy: Node) -> Node:
    """Ensembles `N_POLICY_MEMBERS` copies of `policy` averaged by `jnp.mean`."""
    return ensemble(policy, n=N_POLICY_MEMBERS) >> reduce(jnp.mean)


def pendulum_critic() -> Node:
    """Builds a mean-reduced `PendulumCritic` ensemble initialized on default pendulum state."""
    return (
        ensemble(
            PendulumCritic(ScalarMLP(hidden=HIDDEN)),
            n=N_CRITIC_MEMBERS,
        )
        >> reduce(jnp.mean)
    ).with_input(Pendulum().initialize().state)


def pendulum_training_program(
    policy: Node,
    critic: Node,
    n_episodes: int,
) -> Node:
    """Assembles the complete pendulum SHAC training Node."""
    iteration = shac_iteration(
        policy,
        critic,
        Pendulum(),
        discount=DISCOUNT,
        trace=TRACE,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        n_worlds=N_WORLDS,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
        n_critic_updates=N_CRITIC_UPDATES,
    )
    episode = scan(
        iteration,
        boundary='episode',
        n=N_CHUNKS_PER_EPISODE,
    )
    training = carried(episode)
    data = PendulumTrainingData(
        n_episodes=n_episodes,
        n_chunks_per_episode=N_CHUNKS_PER_EPISODE,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
        n_worlds=N_WORLDS,
        disturbance_scale=DISTURBANCE_SCALE,
    )
    return serial(data=data, training=training)


def train_pendulum_shac(n_episodes: int = N_EPISODES) -> Struct:
    """Trains the recurrent pendulum SHAC policy ensemble and returns the policy, EMA critic, and history."""
    critic = pendulum_critic()
    program = pendulum_training_program(
        policy_committee(pendulum_policy(nn.GRU(MEMORY))),
        critic,
        n_episodes=n_episodes,
    )
    outcome = shac_training(
        program,
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    return Struct(
        policy=outcome.policy,
        critic=critic.bind(outcome.iteration.ema_critic.state),
        history=outcome.history,
    )


def pendulum_evaluation(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    steps: int,
    rng: jax.Array,
) -> Struct:
    """Summarizes episode cost, worst-case terminal state error, and peak torque across `starts`."""
    trajectory = control.policy_trajectory(policy, plant, starts, steps, rng)
    return Struct(
        cost=jnp.mean(jnp.sum(trajectory.cost, axis=1)),
        final_angle=jnp.max(jnp.abs(trajectory.final_state.angle)),
        final_velocity=jnp.max(jnp.abs(trajectory.final_state.velocity)),
        max_torque=jnp.max(jnp.abs(trajectory.action)),
    )


def plot_phase_space(
    terminal_value: PNode,
    trajectory: Struct,
    filename: str = 'pendulum_shac_phase_space_energy.png',
) -> str:
    """Renders closed-loop phase rollouts beside the learned EMA terminal cost surface."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid_state = phase_grid()
    flat_state = jax.tree.map(lambda value: value.reshape(-1), grid_state)
    n_worlds = tree_len(flat_state)

    value = batch(terminal_value, n=n_worlds).apply(flat_state)
    terminal_cost = drop_aux(value).reshape(grid_state.angle.shape)

    figure, (axis, cost_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        sharey=True,
    )
    phase_portrait(axis, grid_state)
    phase_surface(
        cost_axis,
        grid_state,
        terminal_cost,
        label='learned terminal cost',
    )

    references = tree_len(downward_starts())
    reference_state = jax.tree.map(
        lambda value: value[:references],
        trajectory.state,
    )
    random_state = jax.tree.map(
        lambda value: value[references:],
        trajectory.state,
    )
    cyan_references = ('cyan',) * references
    cyan_random = ('cyan',) * random_state.angle.shape[0]
    overlay_phase_trajectories(axis, trajectory.state)
    overlay_trajectories(
        cost_axis,
        reference_state,
        cyan_references,
        linewidth=1.2,
        alpha=0.8,
        mark_starts=False,
    )
    overlay_trajectories(
        cost_axis,
        random_state,
        cyan_random,
        linewidth=0.65,
        alpha=0.32,
        mark_starts=False,
    )
    figure.suptitle(
        'SHAC pendulum control',
        y=0.98,
        fontweight='bold',
    )
    axis.set_title(
        'real closed-loop rollouts from sampled starts, '
        rf'$|\tau| \leq {MAX_TORQUE:g}$',
        color='0.35',
        fontsize=10,
        pad=8,
    )
    cost_axis.set_title(
        'EMA terminal cost',
        color='0.35',
        fontsize=10,
        pad=8,
    )
    axis.legend(loc='upper right', frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = os.path.join(
        os.path.dirname(__file__),
        'plots',
        filename,
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def pendulum_swing_up() -> None:
    """Trains, evaluates, and plots the recurrent SHAC pendulum experiment."""
    result = train_pendulum_shac()
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        downward_starts(),
        steps=N_EVALUATION_STEPS,
        rng=jax.random.PRNGKey(21),
    )
    trajectory = control.policy_trajectory(
        result.policy,
        Pendulum(),
        phase_starts(jax.random.PRNGKey(29)),
        steps=N_EVALUATION_STEPS,
        rng=jax.random.PRNGKey(30),
    )
    portrait = plot_phase_space(result.critic, trajectory)
    print(
        f'final error: {outcome.final_angle:.4f} rad, '
        f'{outcome.final_velocity:.4f} rad/s'
    )
    print(f'mean training cost per step: {result.history.mean_cost[-1]:.4f}')
    print(f'phase portrait: {portrait}')


if __name__ == '__main__':
    pendulum_swing_up()
