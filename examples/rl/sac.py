"""Soft Actor-Critic over a replay Buffer that is ordinary Node state.

The buffer's currency is one chunk per row: an observation sequence with
one step of overlap, the commands and costs taken along it, and the policy
state at the chunk start. A feed-forward policy leaves the stored state
empty and unread, while a recurrent policy resumes from it, so one code
path serves both lifecycles with no fork anywhere. Collection and
chunk-start recovery are the shared machinery in ``control.py``; the
buffer never learns what a row means.

One SAC iteration collects fresh chunks, inserts them, draws every
minibatch of the iteration in one gather, and scans one gradient update
over them. One gradient update is itself a Node whose members are the
actor, critic, and temperature trainers plus the slow target; the buffer
shares the iteration's state with all of them, so ``carried`` over an
injected data Node runs the complete training as one jitted apply.

Sizes are free integers named with their own scope, one training
iteration being the ambient scope: ``n_worlds``, ``n_chunks`` collected,
``n_steps_per_chunk`` (also the truncation depth replay resumes across),
``n_updates``, and ``n_chunks_per_minibatch``, the sample one gradient
step consumes. The policy's ``sample`` method returns the command with
its log-probability. Actor, critic, and temperature update from the
pre-update snapshot of each other's parameters, like the GAN example's
trainers.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
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
    tree_broadcast_axis,
    tree_last,
    tree_reshape,
    tree_tail,
    tree_swap_axes,
    train_step,
)
from nodejax import nn
from examples.rl.control import SamplingStep, chunk_starts, collect
from examples.rl.replay import Buffer


@node
def SampledCommand(policy: Node) -> Node:
    """Replay stored observations, drawing a fresh command at every step.

    A recurrent chunk's starting state arrives as data: init adopts
    ``initial`` verbatim, so replay resumes the memory observed during
    collection; over a stateless policy the declaration is vacuous and
    the field rides unread. It rides every step and is read once.
    """
    def apply(self, observation, initial, rng):
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        return Struct(
            observation=observation,
            command=drawn.command,
            logprob=drawn.logprob,
        )

    def init(input):
        return input.initial

    return Wrapper(policy=policy)(apply, init=init)


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


def entropy_regularized_cost(output: Struct, target: Struct) -> jax.Array:
    """SAC's actor objective: pessimistic cost plus entropy pressure.

    The bound online critic and the temperature arrive as loss data, so
    only the fresh commands carry gradient into the value; that path is
    the actor's signal.
    """
    cost = drop_aux(target.critic.apply(
        Struct(observation=output.observation, command=output.command),
    ))
    return jnp.mean(cost + target.alpha * output.logprob)


def soft_bellman_fit(discount: float, fit: Callable) -> Callable:
    """Configure the critic objective as a two-argument callable.

    The soft Bellman target built from the replayed tail and the bound
    EMA critic arriving as loss data, fitted by ``fit(output, targets)``;
    the targets are constants by capture, so no gradient stopping is
    needed.
    """
    def loss(output, target) -> jax.Array:
        future = drop_aux(target.critic.apply(
            Struct(
                observation=target.tail.observation,
                command=target.tail.command,
            ),
        ))
        targets = target.cost + discount * (
            future + target.alpha * target.tail.logprob)
        return fit(output, targets)

    return loss


@node
def SACUpdate(
    policy: Node,
    critic: Node,
    *,
    transition: PyTree,
    actor_loss: Callable,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    temperature_optimizer,
    n_chunks_per_minibatch: int,
    n_steps_per_chunk: int,
    ema_critic_decay: float,
    target_entropy: float,
    initial_temperature: float,
) -> Node:
    """One SAC gradient update consuming one replayed minibatch of chunks.

    ``transition`` resolves the policy and critic contracts. One replay
    pass over each chunk's overlapping observation sequence serves
    everything: fresh commands and log-probabilities at every step feed
    the actor loss, and the same pass shifted one step supplies the soft
    Bellman target from the EMA critic, a lagging view of the online
    critic parameters advanced once per update and constructed here
    because the warm start assumes its state layout. Both losses receive
    bound critic views as loss data; ``critic_loss`` may inspect Aux, so
    an ensemble critic can fit every member while exposing one
    pessimistic value to the backup.
    """
    policy = policy.with_input(transition.observation)
    critic = critic.with_input(
        Struct(
            observation=transition.observation,
            command=transition.command,
        ),
    )
    ema_critic = nn.EMA(tau=ema_critic_decay, warm=True)
    replay = scanned(
        batch(SampledCommand(policy), n=n_chunks_per_minibatch),
    )
    # One critic tower per shape; each consumer drops the Aux it does
    # not read, so the trainer may fit every member value the ensemble
    # retains while the backup reads the pessimistic value.
    minibatch_critic = batch(
        batch(critic, n=n_chunks_per_minibatch),
        n=n_steps_per_chunk,
        axis='time',
    )
    sequence_critic = batch(
        batch(critic, n=n_chunks_per_minibatch),
        n=n_steps_per_chunk + 1,
        axis='time',
    )

    actor_step = train_step(replay, actor_loss, actor_optimizer)
    critic_step = train_step(minibatch_critic, critic_loss, critic_optimizer)
    temperature = Temperature(initial_temperature)
    temperature_step = train_step(
        temperature,
        temperature_loss(target_entropy),
        temperature_optimizer,
    )
    members = Composite(
        actor_trainer=actor_step,
        critic_trainer=critic_step,
        temperature_trainer=temperature_step,
        ema_critic=ema_critic,
    )

    def apply(self, observation, command, cost, initial, rng):
        alpha = temperature.bind(
            self.state.temperature_trainer.opt.params).value()
        policy_param = self.state.actor_trainer.opt.params

        # One axis move: chunk rows arrive row-major, replay scans time.
        steps = tree_swap_axes(Struct(observation=observation, command=command, cost=cost), 0, 1)
        rider = tree_broadcast_axis(initial, n_steps_per_chunk + 1, axis=0)
        replayed = replay.bind(policy_param).apply(
            observation=steps.observation,
            initial=rider,
            rng=rng.next(),
        )
        ema_critic_param = self.ema_critic(
            self.state.critic_trainer.opt.params)
        # The Bellman target reads the same tensors one step later.
        shifted = tree_tail(Struct(
            observation=steps.observation,
            command=replayed.command,
            logprob=replayed.logprob,
        ))
        self.critic_trainer(
            input=Struct(
                observation=jax.tree.map(
                    lambda value_: value_[:-1], steps.observation),
                command=steps.command,
            ),
            target=Struct(
                cost=steps.cost,
                tail=shifted,
                critic=minibatch_critic.bind(ema_critic_param),
                alpha=alpha,
            ),
        )
        self.actor_trainer(
            input=Struct(observation=steps.observation, initial=rider),
            target=Struct(
                critic=sequence_critic.bind(
                    self.state.critic_trainer.opt.params),
                alpha=alpha,
            ),
        )
        # The Temperature leaf ignores its input; the loss reads the target.
        self.temperature_trainer(
            input=replayed.logprob,
            target=replayed.logprob,
        )
        return Struct(temperature=alpha)

    def init(param):
        """Initialize the trainers and warm-start the target from the
        online critic weights, so the update needs no init input."""
        return Struct(
            actor_trainer=actor_step.bind(param.actor_trainer).init(),
            critic_trainer=critic_step.bind(param.critic_trainer).init(),
            temperature_trainer=temperature_step.bind(
                param.temperature_trainer).init(),
            ema_critic=ema_critic.bind(()).init(
                input=param.critic_trainer.model),
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
    n_chunks: int,
    n_steps_per_chunk: int,
    n_updates: int,
    n_chunks_per_minibatch: int,
) -> Node:
    """One off-policy SAC iteration: collect, store, and update from replay.

    ``update`` is one built gradient-update Node, scanned over the
    iteration's minibatches. ``transition`` is one zero transition fixing
    one step's observation, command, and cost; the stored chunk element
    derives from it and from the policy's own state tree, so the
    iteration constructs without touching the plant.
    """
    policy = policy.with_input(transition.observation)
    memory = jax.tree.map(
        jnp.zeros_like,
        policy.parameterize(rng=jax.random.PRNGKey(0)).init(
            input=transition.observation,
        ),
    )
    element = Struct(
        observation=tile(transition.observation, n_steps_per_chunk + 1),
        command=tile(transition.command, n_steps_per_chunk),
        cost=tile(transition.cost, n_steps_per_chunk),
        initial=memory,
    )

    sampler = scanned(
        scan(
            batch(SamplingStep(policy, plant), n=n_worlds),
            n=n_steps_per_chunk,
        ),
        record=True,
    )
    members = Composite(
        update=scan(update, n=n_updates),
        buffer=Buffer(capacity, element),
    )

    def apply(self, initial_state, disturbance, rng):
        policy_param = self.state.update.actor_trainer.opt.params
        record, recorded_state = collect(
            sampler,
            policy_param,
            initial_state,
            disturbance,
            rng.next(),
            n_chunks=n_chunks,
            n_steps_per_chunk=n_steps_per_chunk,
        )
        starts = chunk_starts(
            policy,
            policy_param,
            record.observation,
            recorded_state,
            n_worlds,
        )

        # One overlap step lets a single replay pass serve both the fresh
        # actions and the shifted Bellman targets. The buffer's currency
        # is one chunk per row: flattening the (chunk, world) axes
        # decorrelates sampling across both.
        sequence = jax.tree.map(
            lambda observation, last: jnp.concatenate(
                (observation, last[:, -1:]), axis=1),
            record.observation,
            record.next_observation,
        )
        rows = tree_swap_axes(
            Struct(
                observation=sequence,
                command=record.command,
                cost=record.cost,
            ),
            1,
            2,
        )
        rows = tree_reshape(rows, (n_chunks * n_worlds,), axes=2)
        self.buffer(Struct(
            observation=rows.observation,
            command=rows.command,
            cost=rows.cost,
            initial=tree_reshape(starts, (n_chunks * n_worlds,), axes=2),
        ))

        # One gather supplies every minibatch of the iteration; all draws
        # see the buffer with this iteration's chunks inserted.
        drawn = self.buffer.sample(
            n_updates * n_chunks_per_minibatch, rng=rng.next())
        minibatches = tree_reshape(
            drawn, (n_updates, n_chunks_per_minibatch))
        outcome = self.update(
            observation=minibatches.observation,
            command=minibatches.command,
            cost=minibatches.cost,
            initial=minibatches.initial,
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
        policy=assembly.policy.bind(final.update.actor_trainer.opt.params),
        critic=assembly.critic.bind(final.update.critic_trainer.opt.params),
        history=Struct(
            actor_loss=aux.training.update.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.update.critic_trainer.loss.reshape(-1),
            temperature=aux.training.temperature,
            mean_cost=aux.training.mean_cost,
        ),
    )