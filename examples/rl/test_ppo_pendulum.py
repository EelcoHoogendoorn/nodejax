"""Focused checks for the runnable recurrent PPO Pendulum example."""

import jax
import jax.numpy as jnp
import numpy as np

from nodejax import (
    Node,
    ensemble,
    reduce,
)
from nodejax import nn
from examples.rl.distributions import (
    LearnedGaussian,
    StateIndependentLogStd,
)
from examples.rl.losses import mse
from examples.rl.ppo import ppo_training
from examples.rl.pendulum import (
    Pendulum,
    PendulumMean,
    downward_starts,
    pendulum_evaluation,
)
from examples.rl.ppo_pendulum import (
    HIDDEN,
    INITIAL_LOG_STD,
    MEMORY,
    PendulumValue,
    pendulum_policy,
    pendulum_training_program,
    pendulum_value,
)


N_VALUE_MEMBERS = 2


def min_ensemble_pendulum_value() -> Node:
    """Build the min-valued ensemble used to test architecture-local loss logic."""
    return (
        ensemble(PendulumValue(hidden=HIDDEN), n=N_VALUE_MEMBERS)
        >> reduce(jnp.min)
    )


def min_ensemble_value_loss(reduced_value, target, *, aux) -> jax.Array:
    """Fit every ensemble member before the min reduction."""
    member_values = aux.reduce_min.population
    return mse(member_values, target[..., None])


def test_ppo_improves_both_policy_lifecycles() -> None:
    """A short run moves rollout cost for feed-forward and recurrent policy."""
    policies = (
        pendulum_policy(nn.identity),
        pendulum_policy(nn.GRU(MEMORY)),
    )
    for policy in policies:
        result = ppo_training(
            pendulum_training_program(
                policy,
                pendulum_value(),
                value_loss=mse,
                iterations=20,
            ),
            parameter_key=jax.random.PRNGKey(0),
            training_key=jax.random.PRNGKey(100),
        )
        history = result.history.mean_cost
        assert np.all(np.isfinite(history))
        assert np.mean(history[-4:]) < 0.95 * np.mean(history[:4]), history


def test_min_ensemble_value_uses_the_same_ppo_program() -> None:
    result = ppo_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            min_ensemble_pendulum_value(),
            value_loss=min_ensemble_value_loss,
            iterations=1,
        ),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )

    assert np.all(np.isfinite(result.history.mean_cost))
    assert np.all(np.isfinite(result.history.critic_loss))


def test_evaluation_accepts_an_aux_emitting_policy() -> None:
    policy = LearnedGaussian(
        ensemble(
            PendulumMean(memory=nn.identity, hidden=HIDDEN),
            n=2,
        ) >> reduce(jnp.mean),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
    )
    observation = Pendulum().initialize().observe()
    policy = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0),
    )

    outcome = pendulum_evaluation(
        policy,
        Pendulum(),
        downward_starts(),
        steps=2,
    )

    assert jnp.isfinite(outcome.mean_cost)
    assert jnp.isfinite(outcome.final_angle)
