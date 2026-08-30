"""Short-Horizon Actor-Critic as a fully injected Node program.

One learner differentiates through a short physical rollout, bootstraps its
tail from a target critic, then fits the online critic to stopped TD-lambda
targets. Scans express physical time, gradient n_chunks_per_episode, critic updates, and
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
    batch,
    drop_aux,
    node,
    scan,
    split_aux,
    state_reinit,
    tile,
    tree_last,
    train_step,
)
from nodejax import nn
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
    *,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    n_worlds: int,
    n_steps_per_chunk: int,
    n_critic_updates: int,
    target_decay: float,
    discount: float,
    trace: float,
) -> Node:
    """One short-horizon policy update and fitted-value update.

    The policy and plant form one controlled transition. The target critic
    is an EMA over the online critic parameters, constructed here because
    the learner's use assumes its state layout; it consumes critic
    parameters before the first trajectory is available, so the plant's
    initial state resolves the critic contract during construction.
    ``critic_loss(output, target)`` returns one scalar and may inspect Aux.
    The optimizer arguments are the transformations consumed by
    ``train_step``.
    """
    critic = critic.with_input(plant.initialize().state)

    # Physical and policy state carry across chunks and reset per episode.
    step = state_reinit(ControlledStep(drop_aux(policy), plant), boundary='episode')
    rollout = scan(batch(step, n=n_worlds), n=n_steps_per_chunk)

    # The bootstrap reads the critic's mean; the trainer's model keeps
    # every member value in Aux, so critic_loss may fit them all.
    value = batch(drop_aux(critic), n=n_worlds)
    trajectory_value = batch(value, n=n_steps_per_chunk, axis='time')
    trajectory_critic = batch(batch(critic, n=n_worlds), n=n_steps_per_chunk, axis='time')

    def policy_loss(output, target_param) -> jax.Array:
        # target_param arrives as loss target data; the gradient reaches the
        # terminal value only through next_state, which is the algorithm.
        trajectory = output
        terminal = value.bind(target_param).apply(
            tree_last(trajectory.next_state),
        )
        discounts = discount ** jnp.arange(n_steps_per_chunk)
        running = jnp.sum(discounts[:, None] * trajectory.cost, axis=0)
        return jnp.mean(running + discount**n_steps_per_chunk * terminal)

    policy_step = train_step(rollout, policy_loss, actor_optimizer)
    critic_step = train_step(trajectory_critic, critic_loss, critic_optimizer)
    members = Composite(
        policy=policy_step,
        critic=scan(critic_step, n=n_critic_updates),
        target=nn.EMA(tau=target_decay, warm=True),
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

        self.critic(
            input=tile(trajectory.state, n_critic_updates),
            target=tile(targets, n_critic_updates),
        )
        mean_cost = jnp.mean(trajectory.cost)
        return Struct(mean_cost=mean_cost), Aux(mean_cost=mean_cost)

    return members(apply)


def shac_training(
    assembly: Struct,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run one assembled SHAC program and bind the trained views.

    ``assembly`` is Struct(program, policy, critic): the composed program
    Node beside the unbound views its final carry binds; the slow critic
    reads back from the EMA state.
    """
    control = assembly.program.parameterize(rng=parameter_key)
    final, aux = split_aux(
        jax.jit(control.apply)(rng=training_key),
    )
    return Struct(
        policy=assembly.policy.bind(final.policy.opt.params.policy),
        critic=assembly.critic.bind(final.target),
        history=Struct(
            policy_loss=aux.training.policy.loss.reshape(-1),
            critic_loss=aux.training.critic.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
