"""Focused integration checks for the runnable Pendulum SAC example."""

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
    N_EVALUATION_STEPS,
    PendulumQ,
    pendulum_critic,
    pendulum_critic_loss,
    pendulum_policy,
    pendulum_training_program,
)


def single_pendulum_critic() -> Node:
    """Build one ordinary critic for the non-ensemble variant check."""
    return PendulumQ(hidden=HIDDEN)


def test_sac_learns_the_swing_up() -> None:
    """A short run of the real assembly brings the pendulum up.

    Eight-step collection costs are dominated by uniformly random starts, so
    the useful check is deterministic evaluation from hanging starts. The
    untrained baseline has mean cost near 3.6 and final angle near 2.5.
    """
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            pendulum_critic(),
            iterations=90,
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

    assert np.all(np.isfinite(trained.history.mean_cost))
    assert np.all(np.isfinite(trained.history.critic_loss))
    assert np.all(trained.history.temperature > 0.0)
    assert outcome.mean_cost < 2.0, outcome
    assert outcome.final_angle < 1.0, outcome


def test_recurrent_sac_is_wired() -> None:
    """The real assembly accepts recurrent policy state throughout replay."""
    trained = sac_training(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            pendulum_critic(),
            iterations=2,
            critic_loss=pendulum_critic_loss,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )

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
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )

    assert np.all(np.isfinite(trained.history.mean_cost))
    assert np.all(np.isfinite(trained.history.critic_loss))
