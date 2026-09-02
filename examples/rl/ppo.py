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
smallest piece replay can evaluate independently, so ``n_steps_per_chunk``
is also the truncation depth. Once advantages have been computed, every
(world, chunk) pair is one independent replay row and all rows are shuffled
together. ``n_minibatches_per_epoch * n_chunks_per_minibatch`` determines
the chunks collected per world, while ``n_worlds * n_chunks_per_minibatch``
determines the replay rows consumed by each actor update.

Rollout records are shaped (n_worlds, n_chunks_per_epoch, n_steps_per_chunk, ...);
gathered update tensors are (n_epochs, n_minibatches_per_epoch,
n_rows_per_minibatch, n_steps_per_chunk, ...), each row replayed from its
recorded start.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    LossFn,
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
    tree_first,
    tree_reshape,
    tree_take,
    train_step,
)
from examples.rl.control import SamplingStep
from examples.rl.losses import discounted_backward_sum


@node
def ReplayStep(policy: Node) -> Node:
    """Re-evaluate one stored transition under the current parameters.

    A recurrent chunk's starting state is a state input, ``initial``: init
    adopts it verbatim, so replay resumes from the memory observed during
    collection, and a run internalized by ``scanned`` takes it once beside
    the sequence.
    """
    def apply(self, observation, command):
        proposal = self.policy(observation)
        return Struct(
            logprob=self.policy.logprob(proposal, command),
            entropy=self.policy.entropy(proposal),
        )

    def init(param, initial):
        return initial

    return Wrapper(policy=policy)(apply, init=init)


@node
def clipped_surrogate(
    *,
    clip: float,  # ratio clip radius around one
    entropy_weight: float,  # weight of the entropy bonus against the surrogate
) -> Node:
    """Configure PPO's clipped actor objective from its recorded rollout.

    Within one actor update, ``output.logprob``, ``output.entropy``,
    ``old_logprob``, and ``advantage`` are shaped (row, time). The objective
    averages both axes. The approximate KL to the recorded policy, the fraction
    of clipped ratios, and the mean entropy ride on aux.
    """
    def apply(output: Struct, old_logprob, advantage):
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


@node
def advantage_estimates(
    *,
    discount: float,  # per-step discount in the advantage recursion
    trace: float,  # GAE trace; 0 is one-step, 1 is Monte Carlo
) -> Node:
    """Generalized advantage estimation for one world's chunked trajectory.

    ``reward``, ``value``, and ``next_value`` are shaped (chunk, time).
    The recursion carries across chunk boundaries, and ``advantage`` and
    ``returns`` retain that shape. World batching belongs outside this Node.
    """
    def apply(reward, value, next_value):
        delta = reward + discount * next_value - value

        def reverse_chunk(next_advantage, chunk_delta):
            advantage = discounted_backward_sum(
                chunk_delta,
                discount * trace,
                next_advantage,
            )
            return advantage[0], advantage

        advantage = jax.lax.scan(
            reverse_chunk,
            jnp.zeros((), dtype=delta.dtype),
            delta,
            reverse=True,
        )[1]
        return Struct(advantage=advantage, returns=advantage + value)

    return Leaf(apply)


def shuffled_minibatches(
    key: jax.Array,
    record: Struct,
    advantages: jax.Array,
    *,
    n_epochs: int,
    n_minibatches_per_epoch: int,
) -> Struct:
    """The actor trainer's bundle: one random order of all replay rows per epoch.

    ``record`` is shaped (world, chunk, time, ...). Its leading axes become one
    row axis before shuffling. The result's sequence leaves are shaped
    (epoch, minibatch, row, time, ...), with recorded starts shaped
    (epoch, minibatch, row, ...).
    """
    chunks = record.replace(
        advantage=advantages,
        policy_state=tree_first(record.policy_state, axis=2),
    )
    rows = tree_reshape(chunks, (-1,), axes=2)
    n_rows_per_epoch = rows.cost.shape[0]
    order = jax.random.permutation(
        key,
        jnp.broadcast_to(jnp.arange(n_rows_per_epoch), (n_epochs, n_rows_per_epoch)),
        axis=1,
        independent=True,
    ).reshape(n_epochs, n_minibatches_per_epoch, -1)
    rows = tree_take(rows, order)
    return Struct(
        observation=rows.observation,
        command=rows.command,
        initial=rows.policy_state,
        old_logprob=rows.logprob,
        advantage=rows.advantage,
    )


@node
def PPO(
    sampler: Node,
    trajectory_value: Node,
    trajectory_advantage: Node,
    actor_trainer: Node,
    critic_trainer: Node,
    *,
    n_epochs: int,  # replays of one collection, the actor trainer's outer scan length
    n_minibatches_per_epoch: int,  # actor updates made during each replay epoch
) -> Node:
    """One fixed-horizon continuing-control PPO iteration.

    Initial-state leaves begin (world, ...). ``disturbance`` and returned-record
    leaves begin (world, chunk, time, ...). ``trajectory_advantage`` maps one
    (chunk, time) trajectory per world to estimates on those same axes.

    ``sampler`` rolls the policy out over chunks and worlds, its policy
    parameters arriving in its ``policy`` field, and records the
    observation, the policy state each step saw, the command, its
    log-probability, the cost, and the next observation.
    ``trajectory_value`` values observations on the rollout's axes, its
    whole parameter tree arriving in its ``value`` field. ``actor_trainer``
    replays shuffled rows from their recorded policy states over
    minibatches and epochs; ``critic_trainer`` fits the value to the
    returns. Both expose what they hold as ``params()``. The output is the
    collected rollout; the mean cost per step rides on aux.
    """
    members = Composite(
        sampler=sampler,
        trajectory_value=trajectory_value,
        trajectory_advantage=trajectory_advantage,
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
        estimates = self.trajectory_advantage(
            reward=-record.cost,
            value=values,
            next_value=next_values,
        )
        advantages = (
            (estimates.advantage - jnp.mean(estimates.advantage))
            / (jnp.std(estimates.advantage) + 1e-8)
        )

        self.actor_trainer(bundle=shuffled_minibatches(
            rng.next(),
            record,
            advantages,
            n_epochs=n_epochs,
            n_minibatches_per_epoch=n_minibatches_per_epoch,
        ))
        self.critic_trainer(input=record.observation, target=estimates.returns)
        return record, Aux(mean_cost=jnp.mean(record.cost))

    return members(apply)


def ppo_learner(
    policy: Node,
    value: Node,
    plant: BaseNode,
    *,
    value_loss: LossFn | BaseNode,
    clip: float,  # ratio clip radius around one
    entropy_weight: float,  # weight of the entropy bonus against the surrogate
    actor_optimizer,  # Optax transformation for the policy parameters
    critic_optimizer,  # Optax transformation for the value parameters
    discount: float,  # per-step discount in the advantage recursion
    trace: float,  # GAE trace; 0 is one-step, 1 is Monte Carlo
    n_worlds: int,
    n_steps_per_chunk: int,  # chunk length, also the replay's truncation depth
    n_epochs: int,  # replays of one collection
    n_minibatches_per_epoch: int,  # actor updates made during each replay epoch
    n_chunks_per_minibatch: int,  # per-world chunk factor in one actor minibatch
    n_critic_passes: int,  # value fits per iteration
) -> Node:
    """Assemble one PPO iteration from a policy, a value, and a plant.

    ``value_loss(value_output, target)`` fits the value to the estimated returns
    and may declare ``aux``. The policy returns proposals and owns ``sample``,
    ``logprob``, and ``entropy`` methods; the plant owns observation and
    transition semantics, and its initial observation resolves both model
    contracts.
    """
    n_chunks_per_epoch = n_minibatches_per_epoch * n_chunks_per_minibatch
    n_rows_per_minibatch = n_worlds * n_chunks_per_minibatch
    observation = plant.initialize().observe()
    policy = policy.with_input(observation)
    value = value.with_input(observation)

    sampler = externalize(
        batch(
            scanned(scan(SamplingStep(policy, plant), n=n_steps_per_chunk)),
            n=n_worlds,
        ),
        'policy',
    )
    # Every optimizer call starts each selected chunk from recorded state. No
    # replay memory survives into the next minibatch or epoch.
    replay = batch(scanned(ReplayStep(policy)), n=n_rows_per_minibatch)
    trajectory_value = batch(
        batch(batch(value, n=n_steps_per_chunk), n=n_chunks_per_epoch),
        n=n_worlds,
    )
    trajectory_advantage = batch(
        advantage_estimates(discount=discount, trace=trace), n=n_worlds)
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
        train_step(trajectory_value, value_loss, critic_optimizer),
        n=n_critic_passes,
    )
    return PPO(
        sampler=sampler,
        trajectory_value=externalize(trajectory_value, field='value'),
        trajectory_advantage=trajectory_advantage,
        actor_trainer=actor_trainer,
        critic_trainer=critic_trainer,
        n_epochs=n_epochs,
        n_minibatches_per_epoch=n_minibatches_per_epoch,
    )


def ppo_training(
    training_program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Execute a complete data-to-learner PPO training composition.

    ``training_program`` is not one ``ppo_learner`` iteration. It must supply a
    sequence of training inputs to ``carried(ppo_learner(...))``, returning the
    learner bound to its final state and emitting per-iteration values under
    ``aux.training``. This function parameterizes and executes that whole Node,
    then returns its trained policy, final learner, and named training history.
    """
    learner, aux = split_aux(
        jax.jit(
            training_program.parameterize(rng=parameter_key).apply,
        )(rng=training_key),
    )
    return Struct(
        policy=learner.actor_trainer.trained().pnode.policy,
        learner=learner,
        history=Struct(
            actor_loss=aux.training.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss.reshape(-1),
            approximate_kl=aux.training.actor_trainer.objective.loss.approximate_kl.reshape(-1),
            clip_fraction=aux.training.actor_trainer.objective.loss.clip_fraction.reshape(-1),
            entropy=aux.training.actor_trainer.objective.loss.entropy.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
