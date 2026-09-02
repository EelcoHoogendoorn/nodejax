"""Run recurrent PPO on the Pendulum and save its phase-space plot."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    BaseNode,
    Composite,
    LossFn,
    Node,
    carried,
    node,
    serial,
    tree_len,
)
from nodejax import nn
from examples.rl.control import mean_rollout
from examples.rl.distributions import LearnedGaussian, StateIndependentLogStd
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
from examples.rl.ppo import ppo_learner, ppo_training


HIDDEN = 64
MEMORY = 32
N_WORLDS = 32
N_STEPS_PER_CHUNK = 16
N_EPOCHS = 4
N_MINIBATCHES_PER_EPOCH = 4
N_CHUNKS_PER_MINIBATCH = 2
N_CHUNKS_PER_EPOCH = N_MINIBATCHES_PER_EPOCH * N_CHUNKS_PER_MINIBATCH
N_CRITIC_PASSES = 4
CLIP = 0.2
DISCOUNT = 0.97
TRACE = 0.95
ENTROPY_WEIGHT = 1e-3
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
INITIAL_LOG_STD = -0.5
N_ITERATIONS = 400
N_EVALUATION_STEPS = 300


@node
def PendulumValue(hidden: int) -> Node:
    """Pendulum state value for the advantage baseline."""
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
        return self.body(self.features(input))

    return members(apply)


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the example policy with an explicit memory lifecycle."""
    mean = PendulumMean(memory=memory, hidden=HIDDEN)
    log_std = StateIndependentLogStd(initial=INITIAL_LOG_STD)
    return LearnedGaussian(mean, log_std)


def pendulum_value() -> Node:
    """Build the example value Node."""
    return PendulumValue(hidden=HIDDEN)


def pendulum_training_program(
    policy: Node,
    value: Node,
    *,
    value_loss: LossFn | BaseNode,
    iterations: int,
) -> Node:
    """Build a complete executable Pendulum PPO training program.

    The returned Node has no ordinary inputs and requires ``rng``. It generates
    ``iterations`` batches of initial states and disturbances, runs one PPO
    learner iteration for each batch, and carries the actor and critic training
    state between iterations. Its ordinary output is the PPO learner bound to
    its final state; per-iteration metrics are retained under ``aux.training``.
    """
    learner_iteration = ppo_learner(
        policy,
        value,
        Pendulum(),
        value_loss=value_loss,
        clip=CLIP,
        entropy_weight=ENTROPY_WEIGHT,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        discount=DISCOUNT,
        trace=TRACE,
        n_worlds=N_WORLDS,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
        n_epochs=N_EPOCHS,
        n_minibatches_per_epoch=N_MINIBATCHES_PER_EPOCH,
        n_chunks_per_minibatch=N_CHUNKS_PER_MINIBATCH,
        n_critic_passes=N_CRITIC_PASSES,
    )
    training_data = PendulumTrainingData(
        iterations,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS_PER_EPOCH,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
    )
    return serial(data=training_data, training=carried(learner_iteration))


def save_figure(figure, filename: str) -> str:
    """Save one example figure beside the other RL plots."""
    import os

    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    return output


def swing_up() -> None:
    """Train the recurrent policy, evaluate it, and save its phase portrait."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = ppo_training(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            pendulum_value(),
            value_loss=mse,
            iterations=N_ITERATIONS,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    history = result.history.mean_cost
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        downward_starts(),
        steps=N_EVALUATION_STEPS,
    )
    print(
        f'training cost {history[0]:.3f} -> {history[-1]:.3f} | '
        f'final angle {outcome.final_angle:.3f} rad | '
        f'final velocity {outcome.final_velocity:.3f} rad/s'
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    phase_portrait(axis, phase_grid())
    plotted_starts = phase_starts(jax.random.PRNGKey(29))
    plotted_rollouts = mean_rollout(
        result.policy,
        Pendulum(),
        plotted_starts,
        jnp.zeros((tree_len(plotted_starts), N_EVALUATION_STEPS)),
    )
    overlay_phase_trajectories(axis, plotted_rollouts.state)
    axis.set_title('recurrent PPO pendulum: real rollouts from sampled starts')
    output = save_figure(figure, 'ppo_pendulum_phase_space.png')
    plt.close(figure)
    print(f'phase plot: {output}')


if __name__ == '__main__':
    swing_up()
