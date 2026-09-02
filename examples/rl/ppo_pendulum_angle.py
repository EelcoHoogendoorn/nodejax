"""Run recurrent PPO from angle-only observations and save its phase-space plot."""

import jax
import jax.numpy as jnp

from nodejax import tree_len
from nodejax import nn
from examples.rl.control import mean_rollout
from examples.rl.distributions import LearnedGaussian, StateIndependentLogStd
from examples.rl.pendulum import (
    AngleFeatures,
    Pendulum,
    PendulumMean,
    downward_starts,
    overlay_phase_trajectories,
    pendulum_evaluation,
    phase_grid,
    phase_portrait,
    phase_starts,
)
from examples.rl.ppo_pendulum import (
    HIDDEN,
    INITIAL_LOG_STD,
    MEMORY,
    N_EVALUATION_STEPS,
    pendulum_training_program,
    pendulum_value,
    ppo_training,
    save_figure,
)


N_ITERATIONS = 1200


def angle_only_swing_up() -> None:
    """Train the partially observed policy, evaluate it, and save its plot."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    policy = LearnedGaussian(
        PendulumMean(
            memory=nn.GRU(MEMORY),
            hidden=HIDDEN,
            features=AngleFeatures(),
        ),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
    )
    result = ppo_training(
        pendulum_training_program(
            policy,
            pendulum_value(),
            iterations=N_ITERATIONS,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        downward_starts(),
        steps=N_EVALUATION_STEPS,
    )
    print(
        f'angle-only evaluation cost {outcome.mean_cost:.3f} | '
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
    axis.set_title('angle-only recurrent PPO: real rollouts from sampled starts')
    output = save_figure(figure, 'ppo_pendulum_angle_phase_space.png')
    plt.close(figure)
    print(f'phase plot: {output}')


if __name__ == '__main__':
    angle_only_swing_up()
