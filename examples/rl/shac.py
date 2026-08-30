"""Short-Horizon Actor-Critic as a fully injected Node program.

One learner differentiates through a short physical rollout, bootstraps its
tail from an EMA critic, then fits the online critic to TD-lambda targets.
Scans express physical time and episode chunks, ``iterated`` the critic
updates, and ``carried`` returns the final optimizer and EMA state. Both
objectives arrive as loss callables, so the constructor owns no objective
content.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Node,
    PNode,
    Struct,
    batch,
    drop_aux,
    iterated,
    node,
    scan,
    split_aux,
    state_reinit,
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


def td_lambda_fit(discount: float, trace: float, fit: Callable) -> Callable:
    """Configure the critic objective as a two-argument callable.

    TD-lambda targets computed from the trajectories and the bound EMA
    critic arriving as loss data, fitted by ``fit(output, targets)``; the
    targets are constants by capture, so no gradient stopping is needed.
    """
    def loss(output, target) -> jax.Array:
        trajectories = target.trajectories
        next_values = drop_aux(target.critic.apply(trajectories.next_state))
        targets = td_lambda(
            trajectories.cost,
            next_values,
            discount=discount,
            trace=trace,
        )
        return fit(output, targets)

    return loss


def bootstrapped_cost(discount: float) -> Callable:
    """Configure SHAC's policy objective as a two-argument callable.

    Discounted running costs of the rollout plus the discounted terminal
    values, read from the bound critic arriving as loss data; the
    gradient reaches the terminal values only through next_state, which
    is the algorithm.
    """
    def loss(output, critic: PNode) -> jax.Array:
        trajectories = output
        terminals = drop_aux(critic.apply(
            tree_last(trajectories.next_state),
        ))
        n_steps = trajectories.cost.shape[0]
        discounts = discount ** jnp.arange(n_steps)
        running_costs = jnp.sum(discounts[:, None] * trajectories.cost, axis=0)
        return jnp.mean(running_costs + discount ** n_steps * terminals)

    return loss


@node
def SHAC(
    policy: Node,
    critic: Node,
    plant: BaseNode,
    *,
    policy_loss: Callable,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    n_worlds: int,
    n_steps_per_chunk: int,
    n_critic_updates: int,
    ema_critic_decay: float,
) -> Node:
    """One short-horizon policy update and fitted-value update.

    The policy and plant form one controlled transition. The EMA critic
    is a lagging view of the online critic parameters, constructed here
    because the learner's use assumes its state layout; it consumes critic
    parameters before the first trajectory is available, so the plant's
    initial state resolves the critic contract during construction.
    ``policy_loss(output, critic)`` and ``critic_loss(output, target)``
    receive bound critic views as loss data. The optimizer arguments are
    the transformations consumed by ``train_step``. The controlled step
    declares ``boundary='episode'``: physical and policy state reset when
    an enclosing scan claims that name, and carry across chunks
    otherwise, so the caller structuring episodes owns the claim.
    """
    critic = critic.with_input(plant.initialize().state)

    # Physical and policy state carry across chunks and reset per episode.
    step = state_reinit(
        ControlledStep(drop_aux(policy), plant), boundary='episode')
    trajectories = scan(batch(step, n=n_worlds), n=n_steps_per_chunk)

    # One critic tower per shape; each consumer drops the Aux it does
    # not read, so the trainer may fit every member value the ensemble
    # retains while the bootstraps read the reduced value.
    critics = batch(critic, n=n_worlds)
    trajectories_critic = batch(critics, n=n_steps_per_chunk, axis='time')

    policy_step = train_step(trajectories, policy_loss, actor_optimizer)
    critic_step = train_step(trajectories_critic, critic_loss, critic_optimizer)
    members = Composite(
        policy_trainer=policy_step,
        critic_trainer=iterated(critic_step, n=n_critic_updates),
        ema_critic=nn.EMA(tau=ema_critic_decay, warm=True),
    )

    def apply(self, disturbance, initial_state):
        ema_critic_param = self.ema_critic(
            self.state.critic_trainer.opt.params)

        # One trainer call does both: it rolls the plant out under the
        # current weights and steps them through that rollout; the
        # returned trajectories are the pre-update forward pass.
        trajectories = self.policy_trainer(
            input=Struct(
                disturbance=disturbance,
                initial_state=initial_state,
            ),
            # The trainer's target slot is loss-time data of any kind,
            # not a regression target: it carries the bound EMA critic
            # that policy_loss applies to the rollout's tail inside the
            # differentiated loss.
            target=critics.bind(ema_critic_param),
        )

        self.critic_trainer(
            input=trajectories.state,
            target=Struct(
                trajectories=trajectories,
                critic=trajectories_critic.bind(ema_critic_param),
            ),
        )

        mean_cost = jnp.mean(trajectories.cost)
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
    Node beside the unbound views its final carry binds; the returned
    critic reads back from the EMA state.
    """
    control = assembly.program.parameterize(rng=parameter_key)
    final, aux = split_aux(
        jax.jit(control.apply)(rng=training_key),
    )
    return Struct(
        policy=assembly.policy.bind(final.policy_trainer.opt.params.policy),
        critic=assembly.critic.bind(final.ema_critic),
        history=Struct(
            policy_loss=aux.training.policy_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )