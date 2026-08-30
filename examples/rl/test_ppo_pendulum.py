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
    drop_aux,
    scan,
    scanned,
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
from examples.rl.pendulum import Pendulum
from examples.rl.ppo import (
    ReplayStep,
    SamplingStep,
    chunk_starts,
    clipped_surrogate,
    collect,
    ppo_program,
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
WORLDS = 32
HORIZON = 128
CHUNK = 16
EPOCHS = 4
MINIBATCHES = 4
CRITIC_PASSES = 4
CLIP = 0.2
DISCOUNT = 0.97
TRACE = 0.95
ENTROPY_WEIGHT = 1e-3
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
INITIAL_LOG_STD = -0.5
EVALUATION_STEPS = 300


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the example policy with an explicit memory lifecycle."""
    mean = PendulumMean(memory=memory, hidden=HIDDEN)
    log_std = StateIndependentLogStd(initial=INITIAL_LOG_STD)
    return LearnedGaussian(mean, log_std)


def pendulum_training_program(policy: Node, iterations: int) -> Node:
    """Assemble every Pendulum and PPO choice at the example boundary."""
    data = PendulumTrainingData(
        iterations,
        worlds=WORLDS,
        horizon=HORIZON,
        chunk=CHUNK,
    )
    value = PendulumValue(hidden=HIDDEN)
    plant = Pendulum()
    return ppo_program(
        policy,
        value,
        plant,
        data,
        actor_loss=clipped_surrogate(
            clip=CLIP,
            entropy_weight=ENTROPY_WEIGHT,
        ),
        critic_loss=mse,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        worlds=WORLDS,
        horizon=HORIZON,
        chunk=CHUNK,
        epochs=EPOCHS,
        minibatches=MINIBATCHES,
        critic_passes=CRITIC_PASSES,
        discount=DISCOUNT,
        trace=TRACE,
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
        scan(batch(SamplingStep(policy, plant), n=WORLDS), n=CHUNK),
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

    count = HORIZON // CHUNK
    data = PendulumTrainingData(
        1,
        worlds=WORLDS,
        horizon=HORIZON,
        chunk=CHUNK,
    ).apply(rng=jax.random.PRNGKey(3))
    initial_state = tree_first(data.initial_state)
    disturbance = tree_first(data.disturbance)
    record, states = collect(
        sampler,
        weights,
        initial_state,
        disturbance,
        jax.random.PRNGKey(1),
    )
    starts = chunk_starts(
        policy,
        weights,
        record.observation,
        states,
        WORLDS,
    )
    replay = batch(
        scanned(batch(ReplayStep(policy), n=WORLDS)),
        n=count,
    ).bind(weights)

    def replayed_logprob(policy_state: PyTree) -> jax.Array:
        bundle = Struct(
            observation=record.observation,
            command=record.command,
            initial=tree_broadcast_axis(policy_state, CHUNK, axis=1),
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
        steps=EVALUATION_STEPS,
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
        jnp.zeros((EVALUATION_STEPS, tree_len(starts))),
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


if __name__ == '__main__':
    swing_up()
