"""Train one recurrent Pendulum policy with SHAC and save its phase plot."""

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
from examples.rl.shac import shac_learner, shac_training


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
    """A replaceable scalar function approximator."""
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
    """Independent episodes split into short gradient chunks."""
    def apply(rng):
        disturbance = disturbance_scale * jax.random.normal(
            rng.next(),
            (n_episodes, n_chunks_per_episode, n_steps_per_chunk, n_worlds),
        )
        initial = Struct(
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
        # Nested scans map every input field. Only episode priming consumes the
        # duplicated initial state.
        initial_state = jax.tree.map(
            lambda value: jnp.broadcast_to(
                value[:, None, None],
                (n_episodes, n_chunks_per_episode, n_steps_per_chunk)
                + value.shape[1:],
            ),
            initial,
        )
        return Struct(
            disturbance=disturbance,
            initial_state=initial_state,
        )

    return Leaf(apply)


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the example policy with an explicit memory lifecycle."""
    return PendulumMean(memory=memory, hidden=HIDDEN)


def policy_committee(policy: Node) -> Node:
    """Choose mean ensembling as the policy architecture."""
    return ensemble(policy, n=N_POLICY_MEMBERS) >> reduce(jnp.mean)


def pendulum_critic() -> Node:
    """Build the mean-valued terminal critic."""
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
    """Assemble the complete Pendulum SHAC Node tree."""
    learner = shac_learner(
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
        learner,
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
    """Train the canonical recurrent Pendulum policy with fixed keys."""
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
        critic=critic.bind(outcome.learner.ema_critic.state),
        history=outcome.history,
    )


def pendulum_evaluation(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    steps: int,
    rng: jax.Array,
) -> Struct:
    trajectory = control.policy_trajectory(policy, plant, starts, steps, rng)
    return Struct(
        cost=jnp.mean(jnp.sum(trajectory.cost, axis=0)),
        final_angle=jnp.max(jnp.abs(trajectory.final_state.angle)),
        final_velocity=jnp.max(jnp.abs(trajectory.final_state.velocity)),
        max_torque=jnp.max(jnp.abs(trajectory.action)),
    )


def plot_phase_space(
    terminal_value: PNode,
    trajectory: Struct,
    filename: str = 'pendulum_shac_phase_space_energy.png',
) -> str:
    """Render real closed-loop rollouts beside the EMA critic."""
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
        lambda value: value[:, :references],
        trajectory.state,
    )
    random_state = jax.tree.map(
        lambda value: value[:, references:],
        trajectory.state,
    )
    cyan_references = ('cyan',) * references
    cyan_random = ('cyan',) * random_state.angle.shape[1]
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
    """Train, evaluate, and plot the canonical Pendulum experiment."""
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
