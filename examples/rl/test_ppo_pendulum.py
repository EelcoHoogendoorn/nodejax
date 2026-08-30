"""Pendulum assembly and checks for the reusable recurrent PPO Nodes.

The algorithm lives in ``ppo.py``. Pendulum policy, value, data, and
evaluation Nodes live in ``ppo_pendulum.py``. This file supplies the runnable
training setup and checks feed-forward and GRU policies against the same PPO
program.

Collection records policy state alongside the rollout. Replaying every chunk
from that ordinary data must reproduce its collection-time log-probabilities.
The full swing-up remains a main-only run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (
    BaseNode,
    Leaf,
    Node,
    PyTree,
    Struct,
    batch,
    carried,
    drop_aux,
    scan,
    scanned,
    serial,
    tree_broadcast_axis,
    tree_first,
    tree_len,
)
from nodejax import nn
from examples.rl.distributions import (
    LearnedGaussian,
    StateIndependentLogStd,
)
from examples.rl.losses import mse
from examples.rl.pendulum import (
    AngleFeatures,
    Pendulum,
    downward_starts,
)
from examples.rl.control import SamplingStep, chunk_starts, collect
from examples.rl.ppo import (
    PPO,
    ReplayStep,
    clipped_surrogate,
    ppo_training,
)
from examples.rl.ppo_pendulum import (
    PendulumMean,
    PendulumTrainingData,
    PendulumValue,
    mean_rollout,
    pendulum_evaluation,
)


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
N_EVALUATION_STEPS = 300


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the example policy with an explicit memory lifecycle."""
    mean = PendulumMean(memory=memory, hidden=HIDDEN)
    log_std = StateIndependentLogStd(initial=INITIAL_LOG_STD)
    return LearnedGaussian(mean, log_std)


def pendulum_training_program(policy: Node, iterations: int) -> Struct:
    """Assemble every Pendulum and PPO choice at the example boundary.

    The iteration consumes every one of its own knobs; the program layer
    composes built Nodes."""
    value = PendulumValue(hidden=HIDDEN)
    iteration = PPO(
        policy,
        value,
        Pendulum(),
        actor_loss=clipped_surrogate(
            clip=CLIP,
            entropy_weight=ENTROPY_WEIGHT,
        ),
        critic_loss=mse,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        n_worlds=N_WORLDS,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
        n_epochs=N_EPOCHS,
        n_minibatches_per_epoch=N_MINIBATCHES_PER_EPOCH,
        n_chunks_per_minibatch=N_CHUNKS_PER_MINIBATCH,
        n_critic_passes=N_CRITIC_PASSES,
        discount=DISCOUNT,
        trace=TRACE,
    )
    data = PendulumTrainingData(
        iterations,
        n_worlds=N_WORLDS,
        n_chunks=N_CHUNKS_PER_EPOCH,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
    )
    return Struct(
        program=serial(data=data, training=carried(iteration)),
        policy=policy,
        value=value,
    )


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


def test_ppo_improves_both_policy_lifecycles() -> None:
    """A short run moves rollout cost for feed-forward and recurrent policy."""
    policies = (
        pendulum_policy(nn.identity),
        pendulum_policy(nn.GRU(MEMORY)),
    )
    for policy in policies:
        result = ppo_training(
            pendulum_training_program(policy, iterations=20),
            parameter_key=jax.random.PRNGKey(0),
            training_key=jax.random.PRNGKey(100),
        )
        history = result.history.mean_cost
        assert np.all(np.isfinite(history))
        assert np.mean(history[-4:]) < 0.95 * np.mean(history[:4]), history


def test_replay_reproduces_the_rollout() -> None:
    """Replay every chunk from recorded state and recover its log-probability."""
    policy = pendulum_policy(nn.GRU(MEMORY))
    plant = Pendulum()
    sampler = scanned(
        scan(batch(SamplingStep(policy, plant), n=N_WORLDS), n=N_STEPS_PER_CHUNK),
        record=True,
    )
    observation = plant.initialize().observe()
    weights = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0),
    ).param
    # The production head starts at zero. Activate its memory path here so
    # post-chunk rather than pre-chunk GRU state cannot pass vacuously.
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
    record, states = collect(
        sampler,
        weights,
        initial_state,
        disturbance,
        jax.random.PRNGKey(1),
        n_chunks=N_CHUNKS_PER_EPOCH,
        n_steps_per_chunk=N_STEPS_PER_CHUNK,
    )
    starts = chunk_starts(
        policy,
        weights,
        record.observation,
        states,
        N_WORLDS,
    )
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
    shifted = replayed_logprob(states.policy)
    assert jnp.allclose(reproduced, record.logprob, atol=1e-5)
    assert jnp.max(jnp.abs(shifted - record.logprob)) > 1e-3


def swing_up() -> None:
    """Run the full training budget and write its phase portrait."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.rl.pendulum import (
        downward_starts,
        overlay_trajectories,
        phase_grid,
        phase_portrait,
    )

    policy = pendulum_policy(nn.GRU(MEMORY))
    result = ppo_training(
        pendulum_training_program(policy, iterations=400),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    history = result.history.mean_cost
    starts = downward_starts()
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        starts,
        steps=N_EVALUATION_STEPS,
    )
    print(
        f'training cost {history[0]:.3f} -> {history[-1]:.3f} | '
        f'final angle {outcome.final_angle:.3f} rad | '
        f'final velocity {outcome.final_velocity:.3f} rad/s'
    )

    grid = phase_grid()
    flat = jax.tree.map(lambda value: value.reshape(-1), grid)
    slice_trajectory = mean_rollout(
        result.policy,
        Pendulum(),
        flat,
        jnp.zeros((1, tree_len(flat))),
    )
    actions = slice_trajectory.action[0].reshape(grid.angle.shape)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    phase_portrait(axis, grid, actions)
    rollouts = mean_rollout(
        result.policy,
        Pendulum(),
        starts,
        jnp.zeros((N_EVALUATION_STEPS, tree_len(starts))),
    )
    overlay_trajectories(axis, rollouts.state)
    axis.set_title(
        'recurrent PPO pendulum: fresh-state policy slice, closed-loop rollouts'
    )
    output = os.path.join(
        os.path.dirname(__file__),
        'plots',
        'ppo_pendulum_phase_space.png',
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)


def angle_only_swing_up() -> None:
    """The partially observed run: the policy reads the angle alone, the
    value keeps the full observation, and memory must recover velocity.
    Takes about four times the fully observed budget; run it with
    ``python -m examples.rl.test_ppo_pendulum angle``."""
    policy = LearnedGaussian(
        PendulumMean(
            memory=nn.GRU(MEMORY),
            hidden=HIDDEN,
            features=AngleFeatures(),
        ),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
    )
    result = ppo_training(
        pendulum_training_program(policy, iterations=1200),
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


if __name__ == '__main__':
    import sys
    if 'angle' in sys.argv[1:]:
        angle_only_swing_up()
    else:
        swing_up()
