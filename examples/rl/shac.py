"""Short-Horizon Actor-Critic as a fully injected Node program.

One learner differentiates through a short physical rollout, bootstraps its
tail from a target critic, then fits the online critic to stopped TD-lambda
targets. Scans express physical time, gradient chunks, critic updates, and
episodes; ``carried`` returns the final optimizer and target state.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Node,
    Struct,
    Wrapper,
    batch,
    carried,
    drop_aux,
    node,
    scan,
    serial,
    split_aux,
    state_reinit,
    tile,
    tree_last,
    tree_stop_gradient,
    train_step,
)
from examples.rl.control import ControlledStep


def td_lambda(
    cost: jax.Array,
    next_value: jax.Array,
    *,
    discount: float,
    trace: float,
) -> jax.Array:
    """TD-lambda targets for a time-major world batch."""
    def step(carry, input):
        current_cost, current_value = input
        target = current_cost + discount * (
            (1.0 - trace) * current_value + trace * carry
        )
        return target, target

    target = jax.lax.scan(
        step,
        next_value[-1],
        (cost, next_value),
        reverse=True,
    )[1]
    return target


@node
def SHAC(
    policy: Node,
    critic: Node,
    plant: BaseNode,
    target: BaseNode,
    *,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    worlds: int,
    horizon: int,
    critic_updates: int,
    discount: float,
    trace: float,
) -> Node:
    """One short-horizon policy update and fitted-value update.

    The policy and plant form one controlled transition. The target Node
    consumes critic parameters before the first trajectory is available, so
    the plant's initial state resolves the critic contract during construction.
    ``critic_loss(output, target)`` returns one scalar and may inspect Aux.
    The optimizer arguments are the transformations consumed by
    ``train_step``.
    """
    critic = critic.with_input(plant.initialize().state)

    # Physical and policy state carry across chunks and reset per episode.
    world = state_reinit(
        ControlledStep(drop_aux(policy), plant),
        boundary='episode',
    )
    window = scan(batch(world, n=worlds), n=horizon)

    # Bootstrap from the clean critic value. The fitted loss receives the
    # complete output, so it may train values retained by an ensemble in Aux.
    critic_value = drop_aux(critic)
    terminal_value = batch(critic_value, n=worlds)
    trajectory_value = batch(terminal_value, n=horizon, axis='time')
    trajectory_population = batch(
        batch(critic, n=worlds),
        n=horizon,
        axis='time',
    )
    discounts = discount ** jnp.arange(horizon)

    def policy_loss(output, target_param) -> jax.Array:
        trajectory = output
        fixed_target = tree_stop_gradient(target_param)
        terminal = terminal_value.bind(fixed_target).apply(
            tree_last(trajectory.next_state),
        )
        running = jnp.sum(discounts[:, None] * trajectory.cost, axis=0)
        return jnp.mean(running + discount**horizon * terminal)

    policy_step = train_step(window, policy_loss, actor_optimizer)
    critic_step = train_step(
        trajectory_population,
        critic_loss,
        critic_optimizer,
    )
    members = Composite(
        policy=policy_step,
        critic=scan(critic_step, n=critic_updates),
        target=target,
    )

    def apply(self, disturbance, initial_state):
        target_param = self.target(self.state.critic.opt.params)
        trajectory = self.policy(
            input=Struct(
                disturbance=disturbance,
                initial_state=initial_state,
            ),
            target=target_param,
        )

        next_value = trajectory_value.bind(target_param).apply(
            trajectory.next_state,
        )
        targets = jax.lax.stop_gradient(
            td_lambda(
                trajectory.cost,
                next_value,
                discount=discount,
                trace=trace,
            )
        )

        observation = tree_stop_gradient(trajectory.state)
        self.critic(
            input=tile(observation, critic_updates),
            target=tile(targets, critic_updates),
        )
        mean_cost = jnp.mean(trajectory.cost)
        return Struct(mean_cost=mean_cost), Aux(mean_cost=mean_cost)

    return members(apply)


def shac_program(
    policy: Node,
    critic: Node,
    plant: BaseNode,
    target: BaseNode,
    data: BaseNode,
    *,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    worlds: int,
    horizon: int,
    critic_updates: int,
    chunks: int,
    discount: float,
    trace: float,
) -> Node:
    """Build a complete SHAC run from injected Nodes and callables."""
    learner = SHAC(
        policy,
        critic,
        plant,
        target,
        critic_loss=critic_loss,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        worlds=worlds,
        horizon=horizon,
        critic_updates=critic_updates,
        discount=discount,
        trace=trace,
    )
    episode = scan(learner, boundary='episode', n=chunks)
    program = carried(episode)

    def apply(self, disturbance, initial_state):
        final = self.program(
            disturbance=disturbance,
            initial_state=initial_state,
        )
        return Struct(
            policy=policy.bind(final.policy.opt.params.policy),
            critic=critic.bind(final.target),
        )

    training = Wrapper(program=program)(apply, name='training')
    return serial(data=data, training=training)


def shac_training(
    program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Parameterize and run an assembled SHAC program once."""
    control = program.parameterize(rng=parameter_key)
    trained, aux = split_aux(
        jax.jit(control.apply)(rng=training_key),
    )
    return Struct(
        policy=trained.policy,
        critic=trained.critic,
        history=Struct(
            policy_loss=aux.training.policy.loss.reshape(-1),
            critic_loss=aux.training.critic.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
