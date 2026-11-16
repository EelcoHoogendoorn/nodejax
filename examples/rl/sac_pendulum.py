"""Trains Soft Actor-Critic on the pendulum and plots closed-loop phase trajectories."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    BaseNode,
    Composite,
    LossFn,
    Node,
    PNode,
    Struct,
    carried,
    ensemble,
    node,
    reduce,
    serial,
    tree_len,
)
from nodejax import nn
from examples.rl.control import mean_rollout
from examples.rl.distributions import (
    SquashedGaussian,
    StateIndependentLogStd,
)
from examples.rl.losses import mse
from examples.rl.pendulum import (
    Pendulum,
    PendulumFeatures,
    PendulumMean,
    PendulumTrainingData,
    downward_starts,
    overlay_phase_trajectories,
    pendulum_evaluation,
    phase_grid,
    phase_portrait,
    phase_starts,
)
from examples.rl.sac import sac_iteration, sac_training


HIDDEN = 64
MEMORY = 32
N_CRITIC_MEMBERS = 2
CAPACITY = 20_000
N_WORLDS = 16
N_CHUNKS = 1
N_STEPS_PER_CHUNK = 8
N_UPDATES = 128
N_CHUNKS_PER_MINIBATCH = 16
DISCOUNT = 0.97
EMA_CRITIC_DECAY = 0.995
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
TEMPERATURE_RATE = 1e-3
INITIAL_LOG_STD = -0.5
INITIAL_TEMPERATURE = 0.1
TARGET_ENTROPY = -1.0
COMMAND_SCALE = 3.0
N_ITERATIONS = 300
N_EVALUATION_STEPS = 300


@node
def PendulumQ(hidden: int) -> Node:
    """Predicts pendulum state-command cost-to-go for the soft Bellman backup."""
    members = Composite(
        features=PendulumFeatures(),
        body=(
            nn.Linear(hidden)
            >> nn.tanh
            >> nn.Linear(hidden)
            >> nn.tanh
            >> nn.Projection()
        ),
    )

    def apply(self, input):
        observed = self.features(input.observation)
        return self.body(jnp.append(observed, input.command))

    return members(apply)


def pendulum_transition() -> Struct:
    """Returns a zero transition fixing the replay row fields and shapes."""
    observation = Struct(angle=jnp.zeros(()), velocity=jnp.zeros(()))
    return Struct(
        observation=observation,
        command=jnp.zeros(()),
        cost=jnp.zeros(()),
    )


def pendulum_policy(memory: BaseNode) -> Node:
    """Builds a tanh-squashed Gaussian policy wrapping `memory` inside `PendulumMean`."""
    return SquashedGaussian(
        PendulumMean(memory=memory, hidden=HIDDEN),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
        scale=COMMAND_SCALE,
    )


def pendulum_critic() -> Node:
    """Builds a twin-critic `PendulumQ` ensemble reduced by `jnp.min`."""
    return (
        ensemble(PendulumQ(hidden=HIDDEN), n=N_CRITIC_MEMBERS)
        >> reduce(jnp.min)
    )


def pendulum_critic_loss(reduced_cost: jax.Array, target: jax.Array, *, aux: Struct) -> jax.Array:
    """Fits every critic ensemble member in `aux.reduce.population` against `target`."""
    member_costs = aux.reduce.population
    return mse(member_costs, target[..., None])


def pendulum_training_program(
    policy: Node,
    critic: Node,
    iterations: int,
    *,
    critic_loss: LossFn | BaseNode,
    capacity: int = CAPACITY,
    n_worlds: int = N_WORLDS,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
    n_updates: int = N_UPDATES,
    n_chunks_per_minibatch: int = N_CHUNKS_PER_MINIBATCH,
) -> Node:
    """Assembles the complete pendulum SAC training Node."""
    iteration = sac_iteration(
        policy,
        critic,
        Pendulum(),
        critic_loss=critic_loss,
        transition=pendulum_transition(),
        capacity=capacity,
        discount=DISCOUNT,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        temperature_optimizer=optax.adam(TEMPERATURE_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        target_entropy=TARGET_ENTROPY,
        initial_temperature=INITIAL_TEMPERATURE,
        n_worlds=n_worlds,
        n_chunks=N_CHUNKS,
        n_steps_per_chunk=n_steps_per_chunk,
        n_updates=n_updates,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
    )
    data = PendulumTrainingData(
        iterations,
        n_worlds=n_worlds,
        n_chunks=N_CHUNKS,
        n_steps_per_chunk=n_steps_per_chunk,
    )
    return serial(data=data, training=carried(iteration))


def plot_phase_space(
    policy: PNode,
    filename: str = 'sac_pendulum_phase_space.png',
    title: str = 'SAC pendulum: real rollouts from sampled starts',
) -> str:
    """Plots closed-loop pendulum rollouts under the deterministic mean of `policy`."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    starts = phase_starts(jax.random.PRNGKey(29))
    trajectories = mean_rollout(
        policy,
        Pendulum(),
        starts,
        jnp.zeros((tree_len(starts), N_EVALUATION_STEPS)),
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    phase_portrait(axis, phase_grid())
    overlay_phase_trajectories(axis, trajectories.state)
    axis.set_title(title)
    figure.tight_layout()

    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def swing_up() -> None:
    """Trains the feed-forward SAC policy, evaluates it on downward starts, and saves its phase portrait."""
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            pendulum_critic(),
            iterations=N_ITERATIONS,
            critic_loss=pendulum_critic_loss,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    outcome = pendulum_evaluation(
        trained.policy,
        Pendulum(),
        downward_starts(),
        steps=N_EVALUATION_STEPS,
    )
    print(
        f'evaluation cost {outcome.mean_cost:.3f} | '
        f'temperature {trained.history.temperature[-1]:.3f} | '
        f'final angle {outcome.final_angle:.3f} rad | '
        f'final velocity {outcome.final_velocity:.3f} rad/s'
    )
    print(f'phase plot: {plot_phase_space(trained.policy)}')


if __name__ == '__main__':
    swing_up()
