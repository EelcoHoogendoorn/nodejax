"""Focused checks for reusable reinforcement-learning distribution Nodes."""

import jax
import jax.numpy as jnp

from nodejax import Leaf
from examples.rl.distributions import LearnedGaussian, StateIndependentLogStd


INITIAL_LOG_STD = -0.5


def test_learned_gaussian_owns_joint_distribution() -> None:
    policy = LearnedGaussian(
        Leaf(lambda input: input),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
    ).with_input(jnp.zeros(3)).parameterize()
    proposal = policy.apply(jnp.zeros(3))
    drawn = policy.sample(proposal, rng=jax.random.PRNGKey(0))

    assert drawn.command.shape == (3,)
    assert drawn.logprob.shape == ()
    assert jnp.allclose(
        drawn.logprob,
        policy.logprob(proposal, drawn.command),
    )
    assert policy.entropy(proposal).shape == ()
    assert jnp.allclose(
        policy.entropy(proposal),
        3.0 * (INITIAL_LOG_STD + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)),
    )
