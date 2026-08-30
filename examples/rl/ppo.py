"""Composable recurrent PPO with recorded Node state as replay data.

Collection composes an injected policy and plant into one transition Node,
batches it over worlds, and scans it over the steps of each chunk and over
chunks. ``record=True`` exposes policy state at chunk boundaries alongside
the rollout. Recurrent replay selects its initial state from that
trajectory and runs a fresh scan.

PPO remains explicit: generalized advantage estimation, chunking,
shuffling, clipping, and the actor and critic schedules are statements
about the algorithm. One PPO iteration is a Node whose trainer members are
scanned over minibatches, epochs, and critic passes. ``carried`` runs that
Node over an injected data Node and returns the final trainer state.

Collection cuts each world's contiguous run into chunks: stored sequence
pieces, each kept beside the policy state at its start. A chunk is the
smallest piece replay can evaluate independently: ``n_steps_per_chunk``
is therefore the truncation depth, whole chunks are the atoms shuffling
gathers (one minibatch is the sample one actor gradient step consumes),
and every epoch replays the one collection, chunk by chunk. Every size
is a free integer named with its own scope, one training iteration
being the ambient scope, so no divisibility constraints exist and
``n_chunks_per_epoch = n_minibatches_per_epoch * n_chunks_per_minibatch``
is also the number of chunks collected.

Rollout records are shaped (n_chunks_per_epoch, n_steps_per_chunk, n_worlds, ...);
recorded policy state is (n_chunks_per_epoch, n_worlds, ...), one snapshot per chunk
boundary; gathered update tensors are (n_epochs, n_minibatches_per_epoch,
n_chunks_per_minibatch, n_steps_per_chunk, n_worlds, ...), replayed
time-major within each chunk after one axis move.
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
    iterated,
    node,
    scan,
    scanned,
    split_aux,
    tree_broadcast_axis,
    tree_last,
    tree_take,
    train_step,
)
from examples.rl.control import SamplingStep, chunk_starts, collect




@node
def ReplayStep(policy: Node) -> Node:
    """Re-evaluate one stored transition under the current parameters.

    A recurrent chunk's starting state arrives as data. Prime adopts
    ``initial`` verbatim so replay resumes from the memory observed during
    collection. The field rides every step of the chunk and is read once.
    """
    def apply(self, observation, command, initial):
        proposal = self.policy(observation)
        return Struct(
            logprob=self.policy.logprob(proposal, command),
            entropy=self.policy.entropy(proposal),
        )

    def init(input):
        return input.initial

    return Wrapper(policy=policy)(apply, init=init)


def clipped_surrogate(
    *,
    clip: float,
    entropy_weight: float,
) -> Callable:
    """Configure PPO's clipped actor loss as a two-argument callable."""
    def loss(output: Struct, target: Struct) -> jax.Array:
        ratio = jnp.exp(output.logprob - target.logprob)
        clipped = jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
        surrogate = jnp.minimum(
            ratio * target.advantage,
            clipped * target.advantage,
        )
        return -(
            jnp.mean(surrogate)
            + entropy_weight * jnp.mean(output.entropy)
        )

    return loss


def advantage_estimates(
    rewards: jax.Array,
    values: jax.Array,
    final_values: jax.Array,
    *,
    discount: float,
    trace: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over chunk-major rollout data.

    Advantages come back normalized for the clipped surrogate; returns
    stay in cost units, being the critic's regression target."""
    shape = rewards.shape
    flat_shape = (shape[0] * shape[1],) + shape[2:]
    rewards = rewards.reshape(flat_shape)
    values = values.reshape(flat_shape)
    next_values = jnp.concatenate([values[1:], final_values[None]])
    deltas = rewards + discount * next_values - values

    def step(carry, input):
        estimate = input + discount * trace * carry
        return estimate, estimate

    advantages = jax.lax.scan(
        step,
        jnp.zeros_like(final_values),
        deltas,
        reverse=True,
    )[1]
    returns = advantages + values
    advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1e-8
    )
    return advantages.reshape(shape), returns.reshape(shape)


@node
def PPO(
    policy: Node,
    value: Node,
    plant: BaseNode,
    *,
    actor_loss: Callable,
    critic_loss: Callable,
    actor_optimizer,
    critic_optimizer,
    n_worlds: int,
    n_steps_per_chunk: int,
    n_epochs: int,
    n_minibatches_per_epoch: int,
    n_chunks_per_minibatch: int,
    n_critic_passes: int,
    discount: float,
    trace: float,
) -> Node:
    """One fixed-horizon continuing-control PPO iteration.

    The policy returns proposals and owns ``sample``, ``logprob``, and
    ``entropy`` methods. The plant owns observation and transition semantics.
    Both loss callables accept ``(output, target)`` and return a scalar. The
    optimizer arguments are the transformations consumed by ``train_step``.
    Recurrent policy initialization is deterministic so the first replay
    chunk can reconstruct the rollout's initial memory exactly. The time
    grid derives from its free factors: every iteration collects
    ``n_minibatches_per_epoch * n_chunks_per_minibatch`` chunks of
    ``n_steps_per_chunk`` steps each.
    """
    n_chunks_per_epoch = n_minibatches_per_epoch * n_chunks_per_minibatch

    # Resolve both model contracts from the plant's real observation shape.
    observation = plant.initialize().observe()
    policy = policy.with_input(observation)
    value = value.with_input(observation)

    # The outer fresh scan records state after each chunk. Shifting that trace
    # supplies the state from which each replay chunk originally started.
    sampler = scanned(
        scan(
            batch(SamplingStep(policy, plant), n=n_worlds),
            n=n_steps_per_chunk,
        ),
        record=True,
    )

    # Every optimizer call starts each selected chunk from recorded state. No
    # replay memory survives into the next minibatch or epoch.
    replay = batch(
        scanned(batch(ReplayStep(policy), n=n_worlds)),
        n=n_chunks_per_minibatch,
    )

    actor_step = train_step(replay, actor_loss, actor_optimizer)

    values = batch(value, n=n_worlds)
    trajectories_value = batch(
        batch(values, n=n_steps_per_chunk, axis='time'),
        n=n_chunks_per_epoch,
        axis='chunk',
    )
    critic_step = train_step(
        trajectories_value,
        critic_loss,
        critic_optimizer,
    )
    members = Composite(
        actor_trainer=scan(
            scan(actor_step, n=n_minibatches_per_epoch),
            n=n_epochs,
        ),
        critic_trainer=iterated(critic_step, n=n_critic_passes),
    )

    def apply(self, initial_state, disturbance, rng):
        policy_param = self.state.actor_trainer.opt.params
        record, recorded_state = collect(
            sampler,
            policy_param,
            initial_state,
            disturbance,
            rng.next(),
            n_chunks=n_chunks_per_epoch,
            n_steps_per_chunk=n_steps_per_chunk,
        )
        starts = chunk_starts(
            policy,
            policy_param,
            record.observation,
            recorded_state,
            n_worlds,
        )

        critic_param = self.state.critic_trainer.opt.params
        value_estimates = trajectories_value.bind(critic_param).apply(
            record.observation)
        final_values = values.bind(critic_param).apply(
            tree_last(tree_last(record.next_observation))
        )
        advantages, returns = advantage_estimates(
            -record.cost,
            value_estimates,
            final_values,
            discount=discount,
            trace=trace,
        )
        # One random order per epoch. Gathering the native chunk axis creates
        # the complete (epoch, minibatch, chunk, time, world) update tensor.
        order = jax.random.permutation(
            rng.next(),
            jnp.broadcast_to(jnp.arange(n_chunks_per_epoch), (n_epochs, n_chunks_per_epoch)),
            axis=1,
            independent=True,
        ).reshape(n_epochs, n_minibatches_per_epoch, n_chunks_per_minibatch)
        rows = tree_take(
            Struct(
                observation=record.observation,
                command=record.command,
                logprob=record.logprob,
                advantage=advantages,
            ),
            order,
        )
        selected_starts = tree_take(starts, order)
        initial = tree_broadcast_axis(selected_starts, n_steps_per_chunk, axis=3)
        self.actor_trainer(
            input=Struct(
                observation=rows.observation,
                command=rows.command,
                initial=initial,
            ),
            target=Struct(
                logprob=rows.logprob,
                advantage=rows.advantage,
            ),
        )

        self.critic_trainer(
            input=record.observation,
            target=returns,
        )
        mean_cost = jnp.mean(record.cost)
        return Struct(mean_cost=mean_cost), Aux(mean_cost=mean_cost)

    return members(apply)


def ppo_training(
    assembly: Struct,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run one assembled PPO program and bind the trained views.

    ``assembly`` is Struct(program, policy, value): the composed program
    Node beside the unbound views its final carry binds.
    """
    control = assembly.program.parameterize(rng=parameter_key)
    final, aux = split_aux(
        jax.jit(control.apply)(rng=training_key),
    )
    return Struct(
        policy=assembly.policy.bind(final.actor_trainer.opt.params),
        value=assembly.value.bind(final.critic_trainer.opt.params),
        history=Struct(
            actor_loss=aux.training.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
