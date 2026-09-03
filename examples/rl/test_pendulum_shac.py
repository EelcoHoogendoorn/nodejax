"""Focused behavior checks for the runnable Pendulum SHAC assembly."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Node,
    Struct,
    scan,
    split_aux,
    state_reinit,
    tree_broadcast_axis,
)
from nodejax import nn
from examples.rl.pendulum import (
    Pendulum,
    PendulumCritic,
)
from examples.rl.shac import shac_iteration, shac_training
from examples.rl.shac_pendulum import (
    ACTOR_RATE,
    CRITIC_RATE,
    DISCOUNT,
    EMA_CRITIC_DECAY,
    HIDDEN,
    MEMORY,
    N_POLICY_MEMBERS,
    TRACE,
    ScalarMLP,
    pendulum_critic,
    pendulum_policy,
    pendulum_training_program,
    policy_committee,
)


def single_pendulum_critic() -> Node:
    """Build one unensembled terminal critic."""
    return PendulumCritic(ScalarMLP(hidden=HIDDEN)).with_input(
        Pendulum().initialize().state,
    )


def test_policy_cyclicity_is_selected_by_its_memory_node() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = pendulum_policy(nn.identity).with_input(state).parameterize(
        rng=jax.random.PRNGKey(0),
    )
    recurrent = pendulum_policy(nn.GRU(MEMORY)).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_command = feedforward.apply(state)
    recurrent, recurrent_output = recurrent.apply(state)

    assert feedforward_command.shape == recurrent_output.shape == ()
    assert feedforward.cyclic is False
    assert recurrent.cyclic is True


def test_policy_ensembles_compile_for_both_policy_lifecycles() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = policy_committee(
        pendulum_policy(nn.identity),
    ).with_input(state).parameterize(rng=jax.random.PRNGKey(0))
    recurrent = policy_committee(
        pendulum_policy(nn.GRU(MEMORY)),
    ).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_output = jax.jit(feedforward.apply)(state)
    recurrent, recurrent_output = jax.jit(recurrent.apply)(state)
    feedforward_command, feedforward_aux = split_aux(feedforward_output)
    recurrent_command, recurrent_aux = split_aux(recurrent_output)

    assert feedforward_command.shape == recurrent_command.shape == ()
    assert feedforward_aux.reduce_mean.population.shape == (N_POLICY_MEMBERS,)
    assert recurrent_aux.reduce_mean.population.shape == (N_POLICY_MEMBERS,)
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
    assert all(value.shape[0] == N_POLICY_MEMBERS for value in recurrent_state)


def test_one_shac_iteration_accepts_both_policy_lifecycles() -> None:
    n_worlds = 2
    n_steps_per_chunk = 4
    initial_plant_state = Struct(
        angle=jnp.asarray((0.4, -0.7)),
        velocity=jnp.asarray((-0.2, 0.3)),
    )
    input = Struct(
        disturbance=jnp.zeros((n_worlds, n_steps_per_chunk)),
        initial_plant_state=tree_broadcast_axis(
            initial_plant_state,
            n_steps_per_chunk,
            axis=1,
        ),
    )

    policies = (
        pendulum_policy(nn.identity),
        pendulum_policy(nn.GRU(MEMORY)),
    )
    for policy in policies:
        iteration = shac_iteration(
            policy_committee(policy),
            pendulum_critic(),
            Pendulum(),
            discount=DISCOUNT,
            trace=TRACE,
            actor_optimizer=optax.adam(ACTOR_RATE),
            critic_optimizer=optax.adam(CRITIC_RATE),
            ema_critic_decay=EMA_CRITIC_DECAY,
            n_worlds=n_worlds,
            n_steps_per_chunk=n_steps_per_chunk,
            n_critic_updates=1,
        )
        control = iteration.with_input(input).parameterize(
            rng=jax.random.PRNGKey(2),
        ).initialize(input=input)
        output = jax.jit(control.apply)(bundle=input)[1]
        trajectories, aux = split_aux(output)

        assert jnp.isfinite(trajectories.cost).all()
        assert jnp.isfinite(aux.mean_cost)
        assert jnp.isfinite(aux.policy_trainer.loss)
        assert jnp.isfinite(aux.critic_trainer.loss).all()


def test_single_critic_uses_the_same_shac_program() -> None:
    result = shac_training(
        pendulum_training_program(
            pendulum_policy(nn.identity),
            single_pendulum_critic(),
            n_episodes=1,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )

    assert jnp.isfinite(result.history.policy_loss).all()
    assert jnp.isfinite(result.history.critic_loss).all()


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
    assert jnp.array_equal(
        whole_output,
        jnp.concatenate((first_output, second_output)),
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

    assert jnp.array_equal(first_output, second_output)
