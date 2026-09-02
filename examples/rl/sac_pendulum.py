"""Train Soft Actor-Critic on Pendulum and save its phase-space plot."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    BaseNode,
    Composite,
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
from examples.rl.sac import sac_learner, sac_training


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
    """Pendulum state-command cost-to-go for the soft Bellman backup."""
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
    """Return one zero transition fixing the replay row fields and shapes."""
    observation = Struct(angle=jnp.zeros(()), velocity=jnp.zeros(()))
    return Struct(
        observation=observation,
        command=jnp.zeros(()),
        cost=jnp.zeros(()),
    )


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the bounded Gaussian policy with an explicit memory lifecycle."""
    return SquashedGaussian(
        PendulumMean(memory=memory, hidden=HIDDEN),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
        scale=COMMAND_SCALE,
    )


def pendulum_critic() -> Struct:
    """Pair the minimum-valued critic ensemble with its memberwise fit."""
    model = (
        ensemble(PendulumQ(hidden=HIDDEN), n=N_CRITIC_MEMBERS)
        >> reduce(jnp.min)
    )

    def fit(reduced_cost, target, *, aux) -> jax.Array:
        member_costs = aux.reduce_min.population
        return mse(member_costs, target[..., None])

    return Struct(model=model, fit=fit)


def pendulum_training_program(
    policy: Node,
    critic: Struct,
    iterations: int,
    *,
    capacity: int = CAPACITY,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
    n_chunks_per_minibatch: int = N_CHUNKS_PER_MINIBATCH,
) -> Node:
    """Assemble the complete Pendulum SAC training Node tree."""
    iteration = sac_learner(
        policy,
        critic,
        Pendulum(),
        transition=pendulum_transition(),
        capacity=capacity,
        discount=DISCOUNT,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        temperature_optimizer=optax.adam(TEMPERATURE_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        target_entropy=TARGET_ENTROPY,
        initial_temperature=INITIAL_TEMPERATURE,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS,
        n_steps_per_chunk=n_steps_per_chunk,
        n_updates=N_UPDATES,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
    )
    data = PendulumTrainingData(
        iterations,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS,
        n_steps_per_chunk=n_steps_per_chunk,
    )
    return serial(data=data, training=carried(iteration))


def plot_phase_space(
    policy: PNode,
    filename: str = 'sac_pendulum_phase_space.png',
    title: str = 'SAC pendulum: real rollouts from sampled starts',
) -> str:
    """Plot real closed-loop trajectories under the trained mean policy."""
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
    """Train the canonical SAC policy, evaluate it, and save its plot."""
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            pendulum_critic(),
            iterations=N_ITERATIONS,
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
