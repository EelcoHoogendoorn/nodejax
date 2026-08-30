"""Soft Actor-Critic over a replay Buffer that is ordinary Node state.

One SAC iteration collects fresh transitions, inserts them into the buffer,
draws every minibatch of the iteration in one gather, and scans one gradient
update over them. One gradient update is itself a Node whose members are the
actor, critic, and temperature trainers plus the slow target; the buffer
shares the iteration's state with all of them, so ``carried`` over an
injected data Node runs the complete training as one jitted apply.

Replay draws single transitions, so this SAC takes a feed-forward policy;
sequence replay from stored recurrent state is the PPO example's machinery.
The actor, critic, and temperature update from the pre-update snapshot of
each other's parameters, like the GAN example's trainers.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    PyTree,
    Struct,
    Wrapper,
    batch,
    drop_aux,
    node,
    scan,
    scanned,
    split_aux,
    tile,
    tree_last,
    tree_reshape,
    train_step,
)
from nodejax import nn
from examples.rl.replay import Buffer


@node
def SamplingStep(policy: Node, plant: BaseNode) -> Node:
    """One exploration transition recorded for the replay buffer."""
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state, rng):
        observation = self.plant.observe()
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        output = self.plant(command=drawn.command, disturbance=disturbance)
        return Struct(
            observation=observation,
            command=drawn.command,
            cost=output.cost,
            next_observation=self.plant.observe(),
        )

    def init(self, input):
        """Start a rollout from caller-supplied plant state."""
        return Struct(plant=input.initial_state)

    return members(apply, init=init)


@node
def SampledCommand(policy: Node) -> Node:
    """Draw a reparameterized command and its log-probability."""
    def apply(self, observation, rng):
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        return Struct(
            observation=observation,
            command=drawn.command,
            logprob=drawn.logprob,
        )

    return Wrapper(policy=policy)(apply)


@node
def Temperature(initial: float) -> Node:
    """A learned entropy temperature, stored as its logarithm.

    ``value`` reads the temperature back from any weights, so callers
    holding trained parameters never touch the stored representation."""
    def param():
        return jnp.log(jnp.asarray(initial))

    def value(param) -> jax.Array:
        return jnp.exp(param)

    def apply(param, input):
        return value(param)

    return Leaf(apply, param=param, methods={'value': value})


def temperature_loss(target_entropy: float) -> Callable:
    """Configure the temperature objective as a two-argument callable.

    The target carries the sampled log-probabilities as data; the gradient
    with respect to the stored logarithm raises the temperature while the
    policy's entropy sits below the target and lowers it above.
    """
    def loss(output: jax.Array, target: jax.Array) -> jax.Array:
        return -jnp.log(output) * (jnp.mean(target) + target_entropy)

    return loss


@node
def SACUpdate(
    policy: Node,
    critic: Node,
    *,
    transition: PyTree,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    temperature_optimizer,
    n_transitions_per_minibatch: int,
    discount: float,
    target_decay: float,
    target_entropy: float,
    initial_temperature: float,
) -> Node:
    """One SAC gradient update consuming one replayed minibatch.

    ``transition`` resolves the policy and critic contracts. The Bellman
    target bootstraps from a slow critic, an EMA over the online critic
    parameters advanced once per update, plus the entropy contribution of
    a fresh policy sample at the next observation. The EMA is constructed
    here because the update's warm start assumes its state layout; only
    its decay is a caller's choice. ``critic_loss`` accepts ``(output,
    target)`` and may inspect Aux, so an ensemble critic can fit every
    member while exposing one pessimistic value to the backup.
    """
    policy = policy.with_input(transition.observation)
    critic = critic.with_input(
        Struct(
            observation=transition.observation,
            command=transition.command,
        ),
    )
    target = nn.EMA(tau=target_decay, warm=True)
    fresh = batch(SampledCommand(policy), n=n_transitions_per_minibatch)
    value = batch(drop_aux(critic), n=n_transitions_per_minibatch)
    minibatch_critic = batch(critic, n=n_transitions_per_minibatch)

    def actor_loss(output: Struct, target: Struct) -> jax.Array:
        # target.critic arrives as loss target data, so only the command
        # carries gradient into the value; that path is the actor's signal.
        cost = value.bind(target.critic).apply(
            Struct(observation=output.observation, command=output.command),
        )
        return jnp.mean(cost + target.alpha * output.logprob)

    actor_step = train_step(fresh, actor_loss, actor_optimizer)
    critic_step = train_step(minibatch_critic, critic_loss, critic_optimizer)
    temperature = Temperature(initial_temperature)
    temperature_step = train_step(
        temperature,
        temperature_loss(target_entropy),
        temperature_optimizer,
    )
    members = Composite(
        actor=actor_step,
        critic=critic_step,
        temperature=temperature_step,
        target=target,
    )

    def apply(self, observation, command, cost, next_observation, rng):
        alpha = temperature.bind(self.state.temperature.opt.params).value()
        policy_param = self.state.actor.opt.params
        next_command = fresh.bind(policy_param).apply(
            observation=next_observation,
            rng=rng.next(),
        )
        target_param = self.target(self.state.critic.opt.params)
        future = value.bind(target_param).apply(
            Struct(
                observation=next_observation,
                command=next_command.command,
            ),
        )
        cost_to_go = jax.lax.stop_gradient(
            cost + discount * (future + alpha * next_command.logprob),
        )

        self.critic(
            input=Struct(observation=observation, command=command),
            target=cost_to_go,
        )
        self.actor(
            input=observation,
            target=Struct(critic=self.state.critic.opt.params, alpha=alpha),
        )
        # The Temperature leaf ignores its input; the loss reads the target.
        self.temperature(
            input=next_command.logprob,
            target=next_command.logprob,
        )
        return Struct(temperature=alpha)

    def init(self):
        """Initialize the trainers and warm-start the target from the
        online critic weights, so the update needs no init input."""
        return Struct(
            actor=actor_step.bind(self.actor).init(),
            critic=critic_step.bind(self.critic).init(),
            temperature=temperature_step.bind(self.temperature).init(),
            target=target.bind(()).init(input=self.critic.model),
        )

    return members(apply, init=init)


@node
def SAC(
    policy: Node,
    update: Node,
    plant: Node,
    *,
    transition: PyTree,
    capacity: int,
    n_worlds: int,
    n_steps_per_world: int,
    n_updates: int,
    n_transitions_per_minibatch: int,
) -> Node:
    """One off-policy SAC iteration: collect, store, and update from replay.

    ``update`` is one built gradient-update Node, scanned over the
    iteration's minibatches. ``transition`` is one zero transition fixing
    the replay element, the fields SamplingStep records; it also resolves
    the policy contract, so the iteration constructs without touching the
    plant.
    """
    if policy.cyclic:
        raise ValueError(
            'single-transition replay requires a feed-forward policy'
        )
    policy = policy.with_input(transition.observation)

    sampler = scanned(batch(SamplingStep(policy, plant), n=n_worlds))
    members = Composite(
        update=scan(update, n=n_updates),
        buffer=Buffer(capacity, transition),
    )

    def apply(self, initial_state, disturbance, rng):
        policy_param = self.state.update.actor.opt.params
        record = sampler.bind(Struct(policy=policy_param)).apply(
            disturbance=disturbance,
            initial_state=tile(initial_state, n_steps_per_world),
            rng=rng.next(),
        )

        # The buffer's currency is one transition per row: flattening the
        # collection axes decorrelates sampling across time and worlds.
        self.buffer(tree_reshape(record, (n_steps_per_world * n_worlds,), axes=2))

        # One gather supplies every minibatch of the iteration; all draws
        # see the buffer with this iteration's transitions inserted.
        drawn = self.buffer.sample(n_updates * n_transitions_per_minibatch, rng=rng.next())
        minibatches = tree_reshape(drawn, (n_updates, n_transitions_per_minibatch))
        outcome = self.update(
            observation=minibatches.observation,
            command=minibatches.command,
            cost=minibatches.cost,
            next_observation=minibatches.next_observation,
        )
        mean_cost = jnp.mean(record.cost)
        return Struct(mean_cost=mean_cost), Aux(
            mean_cost=mean_cost,
            temperature=tree_last(outcome.temperature),
        )

    return members(apply)


def sac_training(
    assembly: Struct,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run one assembled SAC program and bind the trained views.

    ``assembly`` is Struct(program, policy, critic): the composed program
    Node beside the unbound views its final carry binds.
    """
    control = assembly.program.parameterize(rng=parameter_key)
    final, aux = split_aux(
        jax.jit(control.apply)(rng=training_key),
    )
    return Struct(
        policy=assembly.policy.bind(final.update.actor.opt.params),
        critic=assembly.critic.bind(final.update.critic.opt.params),
        history=Struct(
            actor_loss=aux.training.update.actor.loss.reshape(-1),
            critic_loss=aux.training.update.critic.loss.reshape(-1),
            temperature=aux.training.temperature,
            mean_cost=aux.training.mean_cost,
        ),
    )
