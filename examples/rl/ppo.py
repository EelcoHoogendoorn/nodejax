"""Composable recurrent PPO with recorded Node state as replay data.

Collection composes an injected policy and plant into one transition Node,
batches it over worlds, and scans it over the steps of each chunk and over
chunks. The transition records the policy state each step saw, so replay
resumes a recurrent policy from the recorded start of any chunk with a
fresh scan.

PPO remains explicit: generalized advantage estimation, chunking,
shuffling, clipping, and the actor and critic schedules are statements
about the algorithm. One PPO iteration holds a sampler whose policy
parameters arrive as data, a value whose parameters arrive as data, an
actor trainer scanned over minibatches and epochs, and a critic trainer
iterated over passes; the assembly builds all four from a policy, a
value, and a plant with stock transforms.

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
gathered update tensors are (n_epochs, n_minibatches_per_epoch,
n_chunks_per_minibatch, n_steps_per_chunk, n_worlds, ...), replayed
time-major within each chunk after one axis move.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    Struct,
    Wrapper,
    batch,
    externalize,
    iterated,
    node,
    scan,
    scanned,
    split_aux,
    tree_broadcast_axis,
    tree_take,
    train_step,
)
from examples.rl.control import SamplingStep


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


@node
def clipped_surrogate(
    *,
    clip: float,  # ratio clip radius around one
    entropy_weight: float,  # weight of the entropy bonus against the surrogate
) -> Node:
    """Configure PPO's clipped actor objective from its recorded rollout.

    The approximate KL to the recorded policy, the fraction of clipped
    ratios, and the mean entropy ride on aux.
    """
    def apply(
        output: Struct,
        old_logprob: jax.Array,
        advantage: jax.Array,
    ) -> tuple[jax.Array, Aux]:
        log_ratio = output.logprob - old_logprob
        ratio = jnp.exp(log_ratio)
        clipped = jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
        surrogate = jnp.minimum(ratio * advantage, clipped * advantage)
        loss = -(jnp.mean(surrogate) + entropy_weight * jnp.mean(output.entropy))
        return loss, Aux(
            approximate_kl=jnp.mean(-log_ratio),
            clip_fraction=jnp.mean(jnp.abs(ratio - 1.0) > clip),
            entropy=jnp.mean(output.entropy),
        )

    return Leaf(apply)


def advantage_estimates(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    *,
    discount: float,
    trace: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over chunk-major rollout data.

    ``next_values`` values each step's next observation, so the recursion
    runs across chunk boundaries as one contiguous run per world.
    Advantages come back normalized for the clipped surrogate; returns
    stay in cost units, being the critic's regression target."""
    shape = rewards.shape
    flat = (shape[0] * shape[1],) + shape[2:]
    deltas = (rewards + discount * next_values - values).reshape(flat)

    def step(carry, delta):
        estimate = delta + discount * trace * carry
        return estimate, estimate

    advantages = jax.lax.scan(step, jnp.zeros(flat[1:]), deltas, reverse=True)[1]
    returns = advantages.reshape(shape) + values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    return advantages.reshape(shape), returns


def shuffled_minibatches(
    key: jax.Array,
    record: Struct,
    advantages: jax.Array,
    *,
    n_epochs: int,
    n_minibatches_per_epoch: int,
    n_chunks_per_minibatch: int,
) -> Struct:
    """The actor trainer's bundle: one random order of all chunks per epoch,
    cut into minibatches, with each chunk replayed from its recorded start.

    ``record`` is a rollout shaped (chunk, time, world, ...); the result holds
    the replay's three inputs beside its old log-probability and advantage, shaped
    (epoch, minibatch, chunk, time, world, ...).
    """
    n_chunks, n_steps_per_chunk = record.cost.shape[:2]
    order = jax.random.permutation(
        key,
        jnp.broadcast_to(jnp.arange(n_chunks), (n_epochs, n_chunks)),
        axis=1,
        independent=True,
    ).reshape(n_epochs, n_minibatches_per_epoch, n_chunks_per_minibatch)
    chunk_starts = jax.tree.map(lambda value: value[:, 0], record.policy_state)
    rows = tree_take(record.replace(advantage=advantages, policy_state=chunk_starts), order)
    return Struct(
        observation=rows.observation,
        command=rows.command,
        initial=tree_broadcast_axis(rows.policy_state, n_steps_per_chunk, axis=3),
        old_logprob=rows.logprob,
        advantage=rows.advantage,
    )


@node
def PPO(
    sampler: Node,
    trajectory_value: Node,
    actor_trainer: Node,
    critic_trainer: Node,
    *,
    discount: float,  # per-step discount in the advantage recursion
    trace: float,  # GAE trace; 0 is one-step, 1 is Monte Carlo
    n_epochs: int,  # replays of one collection, the actor trainer's outer scan length
    n_minibatches_per_epoch: int,  # the actor trainer's inner scan length
    n_chunks_per_minibatch: int,  # chunks one actor update consumes
) -> Node:
    """One fixed-horizon continuing-control PPO iteration.

    ``sampler`` rolls the policy out over chunks and worlds, its policy
    parameters arriving in its ``policy`` field, and records the
    observation, the policy state each step saw, the command, its
    log-probability, the cost, and the next observation.
    ``trajectory_value`` values observations on the rollout's axes, its
    whole parameter tree arriving in its ``value`` field. ``actor_trainer``
    replays shuffled chunks from their recorded policy states over
    minibatches and epochs; ``critic_trainer`` fits the value to the
    returns. Both expose what they hold as ``params()``. The output is the
    collected rollout; the mean cost per step rides on aux.
    """
    members = Composite(
        sampler=sampler,
        trajectory_value=trajectory_value,
        actor_trainer=actor_trainer,
        critic_trainer=critic_trainer,
    )

    def apply(self, initial_state, disturbance, rng):
        record = self.sampler(
            disturbance=disturbance,
            initial_state=initial_state,
            policy=self.actor_trainer.params(),
        )

        value_param = self.critic_trainer.params()
        values = self.trajectory_value(input=record.observation, value=value_param)
        next_values = self.trajectory_value(input=record.next_observation, value=value_param)
        advantages, returns = advantage_estimates(
            -record.cost, values, next_values, discount=discount, trace=trace)

        self.actor_trainer(bundle=shuffled_minibatches(
            rng.next(),
            record,
            advantages,
            n_epochs=n_epochs,
            n_minibatches_per_epoch=n_minibatches_per_epoch,
            n_chunks_per_minibatch=n_chunks_per_minibatch,
        ))
        self.critic_trainer(input=record.observation, target=returns)
        return record, Aux(mean_cost=jnp.mean(record.cost))

    return members(apply)


def ppo_learner(
    policy: Node,
    value: Struct,
    plant: BaseNode,
    *,
    clip: float,  # ratio clip radius around one
    entropy_weight: float,  # weight of the entropy bonus against the surrogate
    actor_optimizer,  # Optax transformation for the policy parameters
    critic_optimizer,  # Optax transformation for the value parameters
    discount: float,  # per-step discount in the advantage recursion
    trace: float,  # GAE trace; 0 is one-step, 1 is Monte Carlo
    n_worlds: int,
    n_steps_per_chunk: int,  # chunk length, also the replay's truncation depth
    n_epochs: int,  # replays of one collection
    n_minibatches_per_epoch: int,
    n_chunks_per_minibatch: int,  # chunks one actor gradient step consumes
    n_critic_passes: int,  # value fits per iteration
) -> Node:
    """Assemble one PPO iteration from a policy, a value, and a plant.

    ``value`` is Struct(model, fit): the state value Node and the loss that
    fits it to the returns, which may declare ``aux``. The policy returns
    proposals and owns ``sample``, ``logprob``, and ``entropy`` methods; the
    plant owns observation and transition semantics, and its initial
    observation resolves both model contracts.
    """
    n_chunks_per_epoch = n_minibatches_per_epoch * n_chunks_per_minibatch
    observation = plant.initialize().observe()
    policy = policy.with_input(observation)
    value_model = value.model.with_input(observation)

    sampler = externalize(
        scanned(scan(batch(SamplingStep(policy, plant), n=n_worlds), n=n_steps_per_chunk)),
        'policy',
    )
    # Every optimizer call starts each selected chunk from recorded state. No
    # replay memory survives into the next minibatch or epoch.
    replay = batch(scanned(batch(ReplayStep(policy), n=n_worlds)), n=n_chunks_per_minibatch)
    trajectory_value = batch(
        batch(batch(value_model, n=n_worlds), n=n_steps_per_chunk, axis='time'),
        n=n_chunks_per_epoch,
        axis='chunk',
    )
    actor_trainer = scan(
        scan(
            train_step(
                replay,
                clipped_surrogate(clip=clip, entropy_weight=entropy_weight),
                actor_optimizer,
            ),
            n=n_minibatches_per_epoch,
        ),
        n=n_epochs,
    )
    critic_trainer = iterated(
        train_step(trajectory_value, value.fit, critic_optimizer),
        n=n_critic_passes,
    )
    return PPO(
        sampler=sampler,
        trajectory_value=externalize(trajectory_value, field='value'),
        actor_trainer=actor_trainer,
        critic_trainer=critic_trainer,
        discount=discount,
        trace=trace,
        n_epochs=n_epochs,
        n_minibatches_per_epoch=n_minibatches_per_epoch,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
    )


def ppo_training(
    program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run a PPO program and return what it trained.

    ``program`` carries a PPO iteration over its data. The result holds the
    trained policy, read out of the actor trainer's trained model, the
    iteration bound to its final state, from which a caller binds a value
    to the fitted parameters, and the history.
    """
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    return Struct(
        policy=final.actor_trainer.trained().pnode.policy,
        learner=final,
        history=Struct(
            actor_loss=aux.training.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss.reshape(-1),
            approximate_kl=aux.training.actor_trainer.objective.loss.approximate_kl.reshape(-1),
            clip_fraction=aux.training.actor_trainer.objective.loss.clip_fraction.reshape(-1),
            entropy=aux.training.actor_trainer.objective.loss.entropy.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
