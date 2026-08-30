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
    PyTree,
    Struct,
    Wrapper,
    batch,
    node,
    scan,
    scanned,
    split_aux,
    tile,
    tree_broadcast_axis,
    tree_first,
    tree_last,
    train_step,
)


@node
def SamplingStep(policy: Node, plant: Node) -> Node:
    """One on-policy transition: sample, act, record what replay needs."""
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state, rng):
        observation = self.plant.observe()
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        output = self.plant(command=drawn.command, disturbance=disturbance)
        return Struct(
            observation=observation,
            command=drawn.command,
            logprob=drawn.logprob,
            cost=output.cost,
            next_observation=self.plant.observe(),
        )

    def init(self, input):
        """Start a rollout from caller-supplied plant state; a stateless
        policy contributes the empty slot."""
        observation = plant.observe(state=input.initial_state)
        return Struct(
            policy=policy.bind(self.policy).init(input=observation),
            plant=input.initial_state,
        )

    return members(apply, init=init)


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

    if not policy.cyclic:
        # No memory is restored, so the initial field rides along unread.
        return Wrapper(policy=policy)(apply)

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
    reward: jax.Array,
    value: jax.Array,
    final_value: jax.Array,
    *,
    discount: float,
    trace: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over chunk-major rollout data."""
    shape = reward.shape
    flat_shape = (shape[0] * shape[1],) + shape[2:]
    reward = reward.reshape(flat_shape)
    value = value.reshape(flat_shape)
    next_value = jnp.concatenate([value[1:], final_value[None]])
    delta = reward + discount * next_value - value

    def step(carry, input):
        estimate = input + discount * trace * carry
        return estimate, estimate

    advantage = jax.lax.scan(
        step,
        jnp.zeros_like(final_value),
        delta,
        reverse=True,
    )[1]
    returns = advantage + value
    return advantage.reshape(shape), returns.reshape(shape)


def collect(
    sampler: Node,
    policy_param: PyTree,
    initial_state: Struct,
    disturbance: jax.Array,
    key: jax.Array,
    *,
    n_chunks_per_epoch: int,
    n_steps_per_chunk: int,
) -> tuple[Struct, PyTree]:
    """Collect a native ``(chunk, time, world)`` rollout and chunk-end state."""
    starts = tile(tile(initial_state, n_steps_per_chunk), n_chunks_per_epoch)
    output = sampler.bind(Struct(policy=policy_param)).apply(
        disturbance=disturbance,
        initial_state=starts,
        rng=key,
    )
    record, aux = split_aux(output)
    # The sampler is scanned with record=True, so aux carries the sampler
    # state after every chunk; replay starts derive from that trace.
    return record, aux.state


def chunk_starts(
    policy: Node,
    policy_param: PyTree,
    observation: Struct,
    chunk_end_state: PyTree,
    n_worlds: int,
) -> PyTree:
    """Recover each chunk's policy state for deterministic policy init."""
    if not policy.cyclic:
        return ()
    first_observation = tree_first(tree_first(observation))
    batched_policy = batch(policy, n=n_worlds).bind(policy_param)
    first = batched_policy.init(input=first_observation)
    # Recorded state includes empty acyclic slots. Binding removes them so the
    # chunk endings have the same public tree as freshly initialized state.
    ends = batched_policy.bind(state=chunk_end_state.policy).state
    return jax.tree.map(
        lambda initial, ending: jnp.concatenate((initial[None], ending[:-1])),
        first,
        ends,
    )


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
    actor = scan(
        scan(actor_step, n=n_minibatches_per_epoch),
        n=n_epochs,
    )

    terminal_value = batch(value, n=n_worlds)
    trajectory_value = batch(
        batch(terminal_value, n=n_steps_per_chunk, axis='time'),
        n=n_chunks_per_epoch,
        axis='chunk',
    )
    critic_step = train_step(
        trajectory_value,
        critic_loss,
        critic_optimizer,
    )
    members = Composite(
        actor=actor,
        critic=scan(critic_step, n=n_critic_passes),
    )

    def apply(self, initial_state, disturbance, rng):
        policy_param = self.state.actor.opt.params
        record, recorded_state = collect(
            sampler,
            policy_param,
            initial_state,
            disturbance,
            rng.next(),
            n_chunks_per_epoch=n_chunks_per_epoch,
            n_steps_per_chunk=n_steps_per_chunk,
        )
        starts = chunk_starts(
            policy,
            policy_param,
            record.observation,
            recorded_state,
            n_worlds,
        )

        critic_param = self.state.critic.opt.params
        values = trajectory_value.bind(critic_param).apply(record.observation)
        final_value = terminal_value.bind(critic_param).apply(
            tree_last(tree_last(record.next_observation))
        )
        advantage, returns = advantage_estimates(
            -record.cost,
            values,
            final_value,
            discount=discount,
            trace=trace,
        )
        advantage = (advantage - jnp.mean(advantage)) / (
            jnp.std(advantage) + 1e-8
        )

        # One random order per epoch. Gathering the native chunk axis creates
        # the complete (epoch, minibatch, chunk, time, world) update tensor.
        order = jax.random.permutation(
            rng.next(),
            jnp.broadcast_to(jnp.arange(n_chunks_per_epoch), (n_epochs, n_chunks_per_epoch)),
            axis=1,
            independent=True,
        ).reshape(n_epochs, n_minibatches_per_epoch, n_chunks_per_minibatch)
        rows = jax.tree.map(
            lambda value_: value_[order],
            Struct(
                observation=record.observation,
                command=record.command,
                logprob=record.logprob,
                advantage=advantage,
            ),
        )
        selected_starts = jax.tree.map(lambda value_: value_[order], starts)
        initial = tree_broadcast_axis(selected_starts, n_steps_per_chunk, axis=3)
        self.actor(
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

        self.critic(
            input=tile(record.observation, n_critic_passes),
            target=tile(returns, n_critic_passes),
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
        policy=assembly.policy.bind(final.actor.opt.params),
        value=assembly.value.bind(final.critic.opt.params),
        history=Struct(
            actor_loss=aux.training.actor.loss.reshape(-1),
            critic_loss=aux.training.critic.loss.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
