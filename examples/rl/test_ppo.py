"""Focused checks for reusable PPO operations."""

import jax
import jax.numpy as jnp

from nodejax import (
    PyTree,
    Struct,
    batch,
    drop_aux,
    scan,
    scanned,
    tree_broadcast_axis,
    tree_first,
)
from nodejax import nn
from examples.rl.control import SamplingStep
from examples.rl.pendulum import Pendulum, PendulumTrainingData
from examples.rl.ppo import ReplayStep
from examples.rl.ppo_pendulum import (
    MEMORY,
    N_CHUNKS_PER_EPOCH,
    N_STEPS_PER_CHUNK,
    N_WORLDS,
    pendulum_policy,
)


def test_replay_reproduces_the_rollout() -> None:
    """Replay every chunk from recorded state and recover its log-probability."""
    policy = pendulum_policy(nn.GRU(MEMORY))
    plant = Pendulum()
    sampler = scanned(
        scan(batch(SamplingStep(policy, plant), n=N_WORLDS), n=N_STEPS_PER_CHUNK),
    )
    observation = plant.initialize().observe()
    weights = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0),
    ).param
    # The production head starts at zero. Activate its memory path here so
    # a chunk's last recorded GRU state, rather than its first, cannot pass
    # vacuously.
    weights = weights.replace(
        mean=weights.mean.replace(
            command=weights.mean.command.replace(
                projection=weights.mean.command.projection.replace(
                    w=jnp.linspace(
                        -0.1,
                        0.1,
                        weights.mean.command.projection.w.shape[0],
                    ),
                ),
            ),
        ),
    )

    data = PendulumTrainingData(
        1,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS_PER_EPOCH,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
    ).apply(rng=jax.random.PRNGKey(3))
    initial_state = tree_first(data.initial_state)
    disturbance = tree_first(data.disturbance)
    record = drop_aux(sampler.bind(Struct(policy=weights)).apply(
        disturbance=disturbance,
        initial_state=initial_state,
        rng=jax.random.PRNGKey(1),
    ))
    starts = jax.tree.map(lambda value: value[:, 0], record.policy_state)
    ends = jax.tree.map(lambda value: value[:, -1], record.policy_state)
    replay = batch(
        scanned(batch(ReplayStep(policy), n=N_WORLDS)),
        n=N_CHUNKS_PER_EPOCH,
    ).bind(weights)

    def replayed_logprob(policy_state: PyTree) -> jax.Array:
        bundle = Struct(
            observation=record.observation,
            command=record.command,
            initial=tree_broadcast_axis(policy_state, N_STEPS_PER_CHUNK, axis=1),
        )
        return drop_aux(replay.apply(bundle=bundle)).logprob

    reproduced = replayed_logprob(starts)
    shifted = replayed_logprob(ends)
    assert jnp.allclose(reproduced, record.logprob, atol=1e-5)
    assert jnp.max(jnp.abs(shifted - record.logprob)) > 1e-3
