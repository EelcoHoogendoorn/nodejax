"""Focused checks for reusable PPO operations."""

import jax
import jax.numpy as jnp

from nodejax import (
    PyTree,
    Struct,
    batch,
    drop_aux,
    replace_by_path,
    scan,
    scanned,
    tree_first,
    tree_last,
    tree_reshape,
)
from nodejax import nn
from examples.rl.control import SamplingStep
from examples.rl.pendulum import Pendulum, PendulumTrainingData
from examples.rl.ppo import (
    ChunkReplay,
    CommandLikelihood,
    StandardizedAdvantageEstimates,
    advantage_estimates,
)
from examples.rl.ppo_pendulum import (
    MEMORY,
    N_CHUNKS_PER_EPOCH,
    N_STEPS_PER_CHUNK,
    N_WORLDS,
    pendulum_policy,
)


def test_advantage_estimates_one_trajectory() -> None:
    estimator = advantage_estimates(discount=0.5, trace=1.0).parameterize()
    estimates = drop_aux(estimator.apply(
        reward=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        value=jnp.zeros((2, 2)),
        next_value=jnp.zeros((2, 2)),
    ))

    expected = jnp.array([[3.25, 4.5], [5.0, 4.0]])
    assert jnp.allclose(estimates.advantage, expected)
    assert jnp.allclose(estimates.returns, expected)


def test_advantage_standardization_preserves_returns() -> None:
    raw_estimator = batch(advantage_estimates(discount=0.5, trace=1.0), n=2)
    estimator = StandardizedAdvantageEstimates(raw_estimator).parameterize()
    reward = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 4.0], [6.0, 8.0]],
    ])
    estimates = drop_aux(estimator.apply(
        reward=reward,
        value=jnp.zeros_like(reward),
        next_value=jnp.zeros_like(reward),
    ))
    raw_advantage = jnp.array([
        [[3.25, 4.5], [5.0, 4.0]],
        [[6.5, 9.0], [10.0, 8.0]],
    ])
    expected = (
        (raw_advantage - jnp.mean(raw_advantage))
        / (jnp.std(raw_advantage) + 1e-8)
    )

    assert jnp.allclose(estimates.advantage, expected)
    assert jnp.allclose(estimates.returns, raw_advantage)


def test_replay_reproduces_the_rollout() -> None:
    """Replay every chunk from recorded state and recover its log-probability."""
    policy = pendulum_policy(nn.GRU(MEMORY))
    plant = Pendulum()
    sampler = batch(
        scanned(scan(SamplingStep(policy, plant), n=N_STEPS_PER_CHUNK)),
        n=N_WORLDS,
    )
    observation = plant.initialize().observe()
    policy_param = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0),
    ).param
    # The production head starts at zero. Activate its memory path here so
    # a chunk's last recorded GRU state, rather than its first, cannot pass
    # vacuously.
    policy_param = replace_by_path(policy_param, {
        '.mean.command.projection.w': lambda weight: jnp.linspace(
            -0.1, 0.1, weight.shape[0]),
    })

    data = PendulumTrainingData(
        1,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS_PER_EPOCH,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
    ).apply(rng=jax.random.PRNGKey(3))
    initial_plant_state = tree_first(data.initial_plant_state)
    disturbance = tree_first(data.disturbance)
    record = drop_aux(sampler.bind(Struct(policy=policy_param)).apply(
        disturbance=disturbance,
        initial_plant_state=initial_plant_state,
        rng=jax.random.PRNGKey(1),
    ))
    rows = tree_reshape(record, (-1,), axes=2)
    starts = tree_first(rows.policy_state, axis=1)
    ends = tree_last(rows.policy_state, axis=1)
    replay = batch(
        ChunkReplay(scan(CommandLikelihood(policy))),
        n=N_WORLDS * N_CHUNKS_PER_EPOCH,
    ).with_input(bundle=Struct(
        observation=rows.observation, command=rows.command, initial=starts,
    )).bind(policy_param).initialize()

    def replayed_logprob(policy_state: PyTree) -> jax.Array:
        bundle = Struct(
            observation=rows.observation,
            command=rows.command,
            initial=policy_state,
        )
        return drop_aux(replay.apply(bundle=bundle)[1]).logprob

    reproduced = replayed_logprob(starts)
    shifted = replayed_logprob(ends)
    assert jnp.allclose(reproduced, rows.logprob, atol=1e-5)
    assert jnp.max(jnp.abs(shifted - rows.logprob)) > 1e-3
