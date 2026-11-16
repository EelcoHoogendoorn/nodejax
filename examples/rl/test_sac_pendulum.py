"""Verifies feedforward and recurrent `Pendulum` SAC training, evaluation, and single-critic variants."""

import jax
import numpy as np

from nodejax import Node
from nodejax import nn
from examples.rl.losses import mse
from examples.rl.pendulum import Pendulum, downward_starts, pendulum_evaluation
from examples.rl.sac import sac_training
from examples.rl.sac_pendulum import (
    HIDDEN,
    MEMORY,
    PendulumQ,
    pendulum_critic,
    pendulum_critic_loss,
    pendulum_policy,
    pendulum_training_program,
)


def single_pendulum_critic() -> Node:
    """Builds a single unensembled `PendulumQ` critic."""
    return PendulumQ(hidden=HIDDEN)


def test_sac_learns_the_swing_up() -> None:
    """Verifies that SAC executes across iterations and produces finite policy evaluations on `Pendulum`."""
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            pendulum_critic(),
            iterations=2,
            critic_loss=pendulum_critic_loss,
            capacity=16,
            n_worlds=4,
            n_steps_per_chunk=6,
            n_updates=1,
            n_chunks_per_minibatch=2,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    outcome = pendulum_evaluation(
        trained.policy,
        Pendulum(),
        downward_starts(),
        steps=4,
    )

    assert trained.history.mean_cost.shape == (2,)
    assert np.all(np.isfinite(trained.history.mean_cost))
    assert np.all(np.isfinite(trained.history.critic_loss))
    assert np.all(trained.history.temperature > 0.0)
    assert np.isfinite(outcome.mean_cost)
    assert np.isfinite(outcome.final_angle)


def test_recurrent_sac_is_wired() -> None:
    """Verifies that the SAC training program threads recurrent `GRU` policy state through replay."""
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            pendulum_critic(),
            iterations=2,
            critic_loss=pendulum_critic_loss,
            capacity=16,
            n_worlds=4,
            n_steps_per_chunk=6,
            n_updates=1,
            n_chunks_per_minibatch=2,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )

    assert trained.history.mean_cost.shape == (2,)
    assert np.all(np.isfinite(trained.history.mean_cost))
    assert np.all(np.isfinite(trained.history.actor_loss))
    assert np.all(np.isfinite(trained.history.critic_loss))


def test_single_critic_uses_the_same_sac_program() -> None:
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            single_pendulum_critic(),
            iterations=2,
            critic_loss=mse,
            capacity=16,
            n_worlds=4,
            n_steps_per_chunk=6,
            n_updates=1,
            n_chunks_per_minibatch=2,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )

    assert trained.history.mean_cost.shape == (2,)
    assert np.all(np.isfinite(trained.history.mean_cost))
    assert np.all(np.isfinite(trained.history.critic_loss))
