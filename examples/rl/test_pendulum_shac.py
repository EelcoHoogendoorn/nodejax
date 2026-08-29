"""Pendulum assembly and checks for the reusable SHAC Nodes.

The algorithm lives in ``shac.py``. Pendulum policy, critic, data, evaluation,
and plotting support live in ``shac_pendulum.py``. This file supplies every
training choice and checks feed-forward and GRU policies against the same SHAC
program.

The policy and critic are real Node ensembles. Policy commands are averaged
before the plant acts. The critic mean supplies bootstrap values while every
retained member is fitted to the same stopped target.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    BaseNode,
    Node,
    Struct,
    ensemble,
    reduce,
    scan,
    split_aux,
    state_reinit,
    tile,
)
from nodejax import nn
from examples.rl.losses import ensemble_mse
from examples.rl.pendulum import Pendulum, downward_starts, phase_starts
from examples.rl.shac import SHAC, shac_program, shac_training
from examples.rl.shac_pendulum import (
    PendulumCritic,
    PendulumPolicy,
    PendulumTrainingData,
    ScalarMLP,
    pendulum_evaluation,
    plot_phase_space,
    policy_trajectory,
)


DISCOUNT = 0.97
TRACE = 0.95
HIDDEN = 64
MEMORY = 64
POLICY_MEMBERS = 3
CRITIC_MEMBERS = 3
HORIZON = 20
WORLDS = 64
CHUNKS = 15
EPISODES = 40
CRITIC_UPDATES = 4
TARGET_DECAY = 0.995
ACTOR_RATE = 0.001
CRITIC_RATE = 0.001
DISTURBANCE_SCALE = 0.03
EVALUATION_STEPS = 300


def pendulum_policy(memory: BaseNode) -> Node:
    """Build the example policy with an explicit memory lifecycle."""
    return PendulumPolicy(memory=memory, hidden=HIDDEN)


def pendulum_training_program(policy: Node, episodes: int) -> Node:
    """Assemble every Pendulum and SHAC choice at the example boundary."""
    committee = ensemble(policy, n=POLICY_MEMBERS) >> reduce(jnp.mean)
    critic = (
        ensemble(
            PendulumCritic(ScalarMLP(hidden=HIDDEN)),
            n=CRITIC_MEMBERS,
        )
        >> reduce(jnp.mean)
    )
    data = PendulumTrainingData(
        episodes=episodes,
        chunks=CHUNKS,
        horizon=HORIZON,
        worlds=WORLDS,
        disturbance_scale=DISTURBANCE_SCALE,
    )
    return shac_program(
        committee,
        critic,
        Pendulum(),
        nn.EMA(tau=TARGET_DECAY, warm=True),
        data,
        critic_loss=ensemble_mse,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        worlds=WORLDS,
        horizon=HORIZON,
        critic_updates=CRITIC_UPDATES,
        chunks=CHUNKS,
        discount=DISCOUNT,
        trace=TRACE,
    )


def test_policy_cyclicity_is_selected_by_its_memory_node() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = pendulum_policy(nn.identity).with_input(state).parameterize(
        rng=jax.random.PRNGKey(0),
    )
    recurrent = pendulum_policy(nn.GRU(MEMORY)).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_command, feedforward_aux = split_aux(
        feedforward.apply(state),
    )
    recurrent, recurrent_output = recurrent.apply(state)
    recurrent_command, recurrent_aux = split_aux(recurrent_output)

    assert feedforward_command.shape == recurrent_command.shape == ()
    assert feedforward.cyclic is False
    assert feedforward_aux.representation.shape == (HIDDEN,)
    assert recurrent.cyclic is True
    assert recurrent_aux.representation.shape == (MEMORY,)


def test_policy_ensembles_compile_for_both_policy_lifecycles() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = (
        ensemble(pendulum_policy(nn.identity), n=POLICY_MEMBERS)
        >> reduce(jnp.mean)
    ).with_input(state).parameterize(rng=jax.random.PRNGKey(0))
    recurrent = (
        ensemble(pendulum_policy(nn.GRU(MEMORY)), n=POLICY_MEMBERS)
        >> reduce(jnp.mean)
    ).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_output = jax.jit(feedforward.apply)(state)
    recurrent, recurrent_output = jax.jit(recurrent.apply)(state)
    feedforward_command, feedforward_aux = split_aux(feedforward_output)
    recurrent_command, recurrent_aux = split_aux(recurrent_output)

    assert feedforward_command.shape == recurrent_command.shape == ()
    assert feedforward_aux.reduce_mean.population.shape == (POLICY_MEMBERS,)
    assert recurrent_aux.reduce_mean.population.shape == (POLICY_MEMBERS,)
    assert jnp.allclose(
        feedforward_command,
        jnp.mean(feedforward_aux.reduce_mean.population),
    )
    assert jnp.allclose(
        recurrent_command,
        jnp.mean(recurrent_aux.reduce_mean.population),
    )
    recurrent_state = jax.tree.leaves(recurrent.state)
    assert recurrent_state
    assert all(value.shape[0] == POLICY_MEMBERS for value in recurrent_state)


def test_one_shac_update_accepts_both_policy_lifecycles() -> None:
    worlds = 2
    horizon = 4
    initial = Struct(
        angle=jnp.asarray((0.4, -0.7)),
        velocity=jnp.asarray((-0.2, 0.3)),
    )
    input = Struct(
        disturbance=jnp.zeros((horizon, worlds)),
        initial_state=tile(initial, horizon),
    )

    policies = (
        pendulum_policy(nn.identity),
        pendulum_policy(nn.GRU(MEMORY)),
    )
    for policy in policies:
        committee = ensemble(policy, n=POLICY_MEMBERS) >> reduce(jnp.mean)
        critic = (
            ensemble(
                PendulumCritic(ScalarMLP(hidden=HIDDEN)),
                n=CRITIC_MEMBERS,
            )
            >> reduce(jnp.mean)
        )
        learner = SHAC(
            committee,
            critic,
            Pendulum(),
            nn.EMA(tau=TARGET_DECAY, warm=True),
            critic_loss=ensemble_mse,
            actor_optimizer=optax.adam(ACTOR_RATE),
            critic_optimizer=optax.adam(CRITIC_RATE),
            worlds=worlds,
            horizon=horizon,
            critic_updates=1,
            discount=DISCOUNT,
            trace=TRACE,
        )
        control = learner.with_input(input).parameterize(
            rng=jax.random.PRNGKey(2),
        ).initialize(input=input)
        output = jax.jit(control.apply)(bundle=input)[1]
        metric, aux = split_aux(output)

        assert jnp.isfinite(metric.mean_cost)
        assert jnp.isfinite(aux.policy.loss)
        assert jnp.isfinite(aux.critic.loss).all()


def test_recurrent_state_carries_across_chunks_and_resets_at_episode() -> None:
    observations = Struct(
        angle=jnp.linspace(-2.0, 1.0, 8),
        velocity=jnp.linspace(0.5, -0.2, 8),
    )
    first = jax.tree.map(lambda value: value[:3], observations)
    second = jax.tree.map(lambda value: value[3:], observations)
    initial_input = jax.tree.map(lambda value: value[0], observations)
    policy = pendulum_policy(nn.GRU(MEMORY)).with_input(
        initial_input,
    ).parameterize(
        rng=jax.random.PRNGKey(3),
    )
    initial_state = policy.init(input=initial_input)

    whole, whole_output = policy.bind(state=initial_state).scan(observations)
    chunked, first_output = policy.bind(state=initial_state).scan(first)
    chunked, second_output = chunked.scan(second)
    whole_command, whole_aux = split_aux(whole_output)
    first_command, first_aux = split_aux(first_output)
    second_command, second_aux = split_aux(second_output)

    assert jnp.array_equal(
        whole_command,
        jnp.concatenate((first_command, second_command)),
    )
    assert jnp.array_equal(
        whole_aux.representation,
        jnp.concatenate((
            first_aux.representation,
            second_aux.representation,
        )),
    )
    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        whole.state,
        chunked.state,
    ))

    episode = scan(
        state_reinit(
            pendulum_policy(nn.GRU(MEMORY)),
            boundary='episode',
        ),
        boundary='episode',
    ).with_input(observations).parameterize(
        rng=jax.random.PRNGKey(3),
    ).initialize(input=observations)
    episode, first_output = episode.apply(observations)
    episode, second_output = episode.apply(observations)
    first_command, first_aux = split_aux(first_output)
    second_command, second_aux = split_aux(second_output)

    assert jnp.array_equal(first_command, second_command)
    assert jnp.array_equal(
        first_aux.representation,
        second_aux.representation,
    )


def test_shac_swings_up_with_either_policy_lifecycle() -> None:
    feedforward = shac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            episodes=EPISODES,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    recurrent = shac_training(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            episodes=EPISODES,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    feedforward_result = pendulum_evaluation(
        feedforward.policy,
        Pendulum(),
        downward_starts(),
        steps=EVALUATION_STEPS,
    )
    recurrent_result = pendulum_evaluation(
        recurrent.policy,
        Pendulum(),
        downward_starts(),
        steps=EVALUATION_STEPS,
    )

    assert jnp.isfinite(feedforward.history.policy_loss).all()
    assert jnp.isfinite(feedforward.history.critic_loss).all()
    assert jnp.isfinite(recurrent.history.policy_loss).all()
    assert jnp.isfinite(recurrent.history.critic_loss).all()
    for result in (feedforward_result, recurrent_result):
        assert result.final_angle < 0.1
        assert result.final_velocity < 0.1


if __name__ == '__main__':
    feedforward = shac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            episodes=EPISODES,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    recurrent = shac_training(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            episodes=EPISODES,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    feedforward_result = pendulum_evaluation(
        feedforward.policy,
        Pendulum(),
        downward_starts(),
        steps=EVALUATION_STEPS,
    )
    recurrent_result = pendulum_evaluation(
        recurrent.policy,
        Pendulum(),
        downward_starts(),
        steps=EVALUATION_STEPS,
    )
    print(
        pendulum_training_program(
            pendulum_policy(nn.GRU(MEMORY)),
            episodes=EPISODES,
        ).describe()
    )
    print(
        'feed-forward final error: '
        f'{feedforward_result.final_angle:.4f} rad, '
        f'{feedforward_result.final_velocity:.4f} rad/s',
    )
    print(
        'recurrent final error: '
        f'{recurrent_result.final_angle:.4f} rad, '
        f'{recurrent_result.final_velocity:.4f} rad/s',
    )
    print(
        'mean training cost per step: '
        f'{feedforward.history.mean_cost[-1]:.4f} feed-forward, '
        f'{recurrent.history.mean_cost[-1]:.4f} recurrent',
    )
    trajectory = policy_trajectory(
        recurrent.policy,
        Pendulum(),
        phase_starts(jax.random.PRNGKey(29)),
        steps=EVALUATION_STEPS,
    )
    portrait = plot_phase_space(
        recurrent.policy,
        recurrent.critic,
        trajectory,
        Pendulum(),
    )
    print(f'phase portrait: {portrait}')
