"""Train recurrent SAC from angle-only observations and save its phase-space plot."""

import jax

from nodejax import nn
from examples.rl.distributions import (
    SquashedGaussian,
    StateIndependentLogStd,
)
from examples.rl.pendulum import (
    AngleFeatures,
    Pendulum,
    PendulumMean,
    downward_starts,
    pendulum_evaluation,
)
from examples.rl.sac_pendulum import (
    COMMAND_SCALE,
    HIDDEN,
    INITIAL_LOG_STD,
    MEMORY,
    N_EVALUATION_STEPS,
    pendulum_critic,
    pendulum_critic_loss,
    pendulum_training_program,
    plot_phase_space,
    sac_training,
)


N_ITERATIONS = 2400
CAPACITY = 2_000
N_STEPS_PER_CHUNK = 16
N_CHUNKS_PER_MINIBATCH = 8


def angle_only_swing_up() -> None:
    """Train the partially observed recurrent SAC policy.

    The policy reads angle alone, while the Q function keeps the full
    observation and stored chunk starts carry the memory replay resumes from.
    Measured on 2026-08-30, this run needs about eight times the fully observed
    budget, a small buffer, and longer chunks.
    """
    policy = SquashedGaussian(
        PendulumMean(
            memory=nn.GRU(MEMORY),
            hidden=HIDDEN,
            features=AngleFeatures(),
        ),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
        scale=COMMAND_SCALE,
    )
    trained = sac_training(
        pendulum_training_program(
            policy,
            pendulum_critic(),
            iterations=N_ITERATIONS,
            critic_loss=pendulum_critic_loss,
            capacity=CAPACITY,
            n_steps_per_chunk=N_STEPS_PER_CHUNK,
            n_chunks_per_minibatch=N_CHUNKS_PER_MINIBATCH,
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
        f'angle-only evaluation cost {outcome.mean_cost:.3f} | '
        f'temperature {trained.history.temperature[-1]:.3f} | '
        f'final angle {outcome.final_angle:.3f} rad | '
        f'final velocity {outcome.final_velocity:.3f} rad/s'
    )
    plot_path = plot_phase_space(
        trained.policy,
        filename='sac_pendulum_angle_phase_space.png',
        title='Angle-only SAC pendulum: real rollouts from sampled starts',
    )
    print(f'phase plot: {plot_path}')


if __name__ == '__main__':
    angle_only_swing_up()
