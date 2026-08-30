"""Pendulum assembly and checks for the off-policy SAC Nodes.

The algorithm lives in ``sac.py`` and the replay buffer in ``replay.py``.
Pendulum Q, transition, and data Nodes live in ``sac_pendulum.py``. The
policy and its mean network are the PPO example's, unchanged: off-policy
training swaps the learner around the same plant and policy family.

The buffer checks pin its cyclic overwrite and the fact that an insert is
visible to a sample inside the same enclosing apply. The training check runs
the full machinery briefly; the swing-up remains a main-only run:
``python -m examples.rl.test_sac_pendulum``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (Composite, Node, Struct, carried, ensemble, node,
                     reduce, serial)
from nodejax import nn
from examples.rl.distributions import (
    SquashedGaussian,
    StateIndependentLogStd,
)
from examples.rl.losses import ensemble_min_mse
from examples.rl.pendulum import Pendulum, downward_starts
from examples.rl.ppo_pendulum import (
    PendulumMean,
    mean_rollout,
    pendulum_evaluation,
)
from examples.rl.replay import Buffer
from examples.rl.sac import SAC, SACUpdate, sac_training
from examples.rl.sac_pendulum import (
    PendulumQ,
    PendulumTrainingData,
    pendulum_transition,
)


HIDDEN = 64
N_CRITIC_MEMBERS = 2
CAPACITY = 20_000
N_WORLDS = 16
N_STEPS_PER_WORLD = 8
N_UPDATES = 128
N_TRANSITIONS_PER_MINIBATCH = 128
DISCOUNT = 0.97
TARGET_DECAY = 0.995
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
TEMPERATURE_RATE = 1e-3
INITIAL_LOG_STD = -0.5
INITIAL_TEMPERATURE = 0.1
TARGET_ENTROPY = -1.0
COMMAND_SCALE = 3.0
N_EVALUATION_STEPS = 300


def pendulum_policy() -> Node:
    """A squashed Gaussian over the PPO example's mean network: bounded
    commands keep the learned Q from dragging the actor into saturation."""
    return SquashedGaussian(
        PendulumMean(memory=nn.identity, hidden=HIDDEN),
        StateIndependentLogStd(initial=INITIAL_LOG_STD),
        scale=COMMAND_SCALE,
    )


def pendulum_training_program(iterations: int) -> Struct:
    """Assemble every Pendulum and SAC choice at the example boundary.

    Each constructor consumes the configuration it owns; the layers
    exchange built Nodes. ``transition`` and ``n_transitions_per_minibatch``
    reach two constructors because two components genuinely consume them."""
    policy = pendulum_policy()
    critic = (
        ensemble(PendulumQ(hidden=HIDDEN), n=N_CRITIC_MEMBERS)
        >> reduce(jnp.min)
    )
    transition = pendulum_transition()
    update = SACUpdate(
        policy,
        critic,
        transition=transition,
        critic_loss=ensemble_min_mse,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        temperature_optimizer=optax.adam(TEMPERATURE_RATE),
        n_transitions_per_minibatch=N_TRANSITIONS_PER_MINIBATCH,
        discount=DISCOUNT,
        target_decay=TARGET_DECAY,
        target_entropy=TARGET_ENTROPY,
        initial_temperature=INITIAL_TEMPERATURE,
    )
    iteration = SAC(
        policy,
        update,
        Pendulum(),
        transition=transition,
        capacity=CAPACITY,
        n_worlds=N_WORLDS,
        n_steps_per_world=N_STEPS_PER_WORLD,
        n_updates=N_UPDATES,
        n_transitions_per_minibatch=N_TRANSITIONS_PER_MINIBATCH,
    )
    data = PendulumTrainingData(
        iterations,
        n_worlds=N_WORLDS,
        n_steps_per_world=N_STEPS_PER_WORLD,
    )
    return Struct(
        program=serial(data=data, training=carried(iteration)),
        policy=policy,
        critic=critic,
    )


def test_buffer_wraps_and_samples_only_filled_rows() -> None:
    capacity = 8
    element = Struct(observation=jnp.zeros(2), cost=jnp.zeros(()))
    buffer = Buffer(capacity, element).parameterize().initialize()
    segment = Struct(
        observation=jnp.arange(10.0).reshape(5, 2),
        cost=jnp.arange(1.0, 6.0),
    )
    buffer, fill = buffer.apply(segment)
    drawn = buffer.sample(64, rng=jax.random.PRNGKey(0))

    assert fill == 5
    assert jnp.all(drawn.cost >= 1.0)
    assert jnp.all(drawn.cost <= 5.0)

    buffer, fill = buffer.apply(jax.tree.map(lambda value: -value, segment))
    wrapped = jnp.array([5, 6, 7, 0, 1])

    assert fill == capacity
    assert jnp.array_equal(buffer.state.store.cost[wrapped], -segment.cost)


@node
def InsertThenSample(buffer: Node, count: int) -> Node:
    """Insert a segment and draw from the same buffer in one apply."""
    members = Composite(buffer=buffer)

    def apply(self, segment, rng):
        fill = self.buffer(segment)
        drawn = self.buffer.sample(count, rng=rng.next())
        return Struct(fill=fill, drawn=drawn)

    return members(apply)


def test_buffer_insert_is_visible_to_a_sample_in_the_same_apply() -> None:
    element = Struct(cost=jnp.zeros(()))
    trip = InsertThenSample(
        Buffer(8, element),
        count=16,
    ).parameterize().initialize()
    segment = Struct(cost=jnp.arange(1.0, 6.0))
    trip, output = jax.jit(trip.apply)(segment, rng=jax.random.PRNGKey(1))

    assert output.fill == 5
    assert jnp.all(output.drawn.cost >= 1.0)


def test_sac_learns_the_swing_up() -> None:
    """A short run of the full machinery brings the pendulum up.

    The collection cost history stays flat by construction: eight-step
    rollouts from uniformly random starts are dominated by their starts.
    The check is the deterministic evaluation from hanging starts, whose
    untrained baseline sits near cost 3.6 and final angle 2.5; at this
    budget an occasional seed leaves one start settling near 0.7 rad, so
    the angle pin separates learned from unlearned, not luck from luck."""
    result = sac_training(
        pendulum_training_program(iterations=60),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        downward_starts(),
        steps=N_EVALUATION_STEPS,
    )

    assert np.all(np.isfinite(result.history.mean_cost))
    assert np.all(np.isfinite(result.history.critic_loss))
    assert np.all(result.history.temperature > 0.0)
    assert outcome.mean_cost < 2.0, outcome
    assert outcome.final_angle < 1.0, outcome


def swing_up() -> None:
    """Run the full training budget and write its phase portrait."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nodejax import tree_len
    from examples.rl.pendulum import (
        overlay_trajectories,
        phase_grid,
        phase_portrait,
    )

    result = sac_training(
        pendulum_training_program(iterations=300),
        parameter_key=jax.random.PRNGKey(0),
        training_key=jax.random.PRNGKey(100),
    )
    starts = downward_starts()
    outcome = pendulum_evaluation(
        result.policy,
        Pendulum(),
        starts,
        steps=N_EVALUATION_STEPS,
    )
    print(
        f'evaluation cost {outcome.mean_cost:.3f} | '
        f'temperature {result.history.temperature[-1]:.3f} | '
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
        'SAC pendulum: mean-policy slice, closed-loop rollouts'
    )
    output = os.path.join(
        os.path.dirname(__file__),
        'plots',
        'sac_pendulum_phase_space.png',
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    swing_up()