"""Implements recurrent PPO with recorded policy state as chunk-level replay data.

Composes an injected policy and plant into `SamplingStep`, batches over worlds,
and scans over steps per chunk and chunks per iteration. Each transition records
the carried policy state at its start so `ChunkReplay` can resume a recurrent
policy from the start of any shuffled `(world, chunk)` row with a fresh scan.

Builds each `PPOIteration` from an externalized rollout sampler, an externalized
value evaluator, GAE advantage estimation, an actor `train_step` scanned over
minibatches and epochs, and a critic `train_step` iterated over passes.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    LossFn,
    Node,
    PNode,
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
from examples.rl.control import SamplingStep, initial_observation
from examples.rl.losses import discounted_backward_sum


@node
def CommandLikelihood(policy: BaseNode) -> Node | PNode:
    """Evaluates log-probability and entropy of a stored `command` under `policy`."""
    def apply(self, observation, command):
        proposal = self.policy(observation)
        return Struct(
            logprob=self.policy.logprob(proposal, command),
            entropy=self.policy.entropy(proposal),
        )

    return Wrapper(policy=policy)(apply)


@node
def ChunkReplay(run: BaseNode) -> Node | PNode:
    """Replays one stored chunk by binding `run` state to the recorded `initial` policy state."""
    def apply(self, observation, command, initial):
        return self.run.bind(state=initial)(observation=observation, command=command)

    return Wrapper(run=run)(apply)


@node
def clipped_surrogate(
    *,
    clip: float,  # ratio clip radius around one
    entropy_weight: float,  # weight of the entropy bonus against the surrogate
) -> PNode:
    """Computes PPO's clipped surrogate loss and entropy bonus over `(row, time)` minibatches."""
    def apply(output: Struct, old_logprob: jax.Array, advantage: jax.Array):
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
    trace: float,  # GAE trace where 0 is one-step and 1 is Monte Carlo
) -> PNode:
    """Computes generalized advantage estimates and returns across `(chunk, time)` for one world."""
    def apply(reward: jax.Array, value: jax.Array, next_value: jax.Array):
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


@node
def StandardizedAdvantageEstimates(estimator: BaseNode) -> Node | PNode:
    """Standardizes `advantage` across the full batch while leaving critic `returns` unscaled."""
    def apply(self, reward, value, next_value):
        estimates = self.estimator(reward=reward, value=value, next_value=next_value)
        advantage = (
            (estimates.advantage - jnp.mean(estimates.advantage))
            / (jnp.std(estimates.advantage) + 1e-8)
        )
        return estimates.replace(advantage=advantage)

    return Wrapper(estimator=estimator)(apply)


def shuffled_minibatches(
    key: jax.Array,
    record: Struct,
    advantages: jax.Array,
    *,
    n_epochs: int,
    n_minibatches_per_epoch: int,
) -> Struct:
    """Flattens `(world, chunk)` rollout chunks into rows and shuffles them independently per epoch into `(epoch, minibatch, row, ...)` bundles."""
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
def PPOIteration(
    sampler: BaseNode,
    trajectory_value: BaseNode,
    trajectory_advantage: BaseNode,
    actor_trainer: BaseNode,
    critic_trainer: BaseNode,
    *,
    n_epochs: int,
    n_minibatches_per_epoch: int,
) -> Node:
    """Collects one chunked on-policy rollout, computes GAE targets, and updates actor and critic trainers."""
    members = Composite(
        sampler=sampler,
        trajectory_value=trajectory_value,
        trajectory_advantage=trajectory_advantage,
        actor_trainer=actor_trainer,
        critic_trainer=critic_trainer,
    )

    def apply(self, initial_plant_state, disturbance, rng):
        record = self.sampler(
            disturbance=disturbance,
            initial_plant_state=initial_plant_state,
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

        self.actor_trainer(bundle=shuffled_minibatches(
            rng.next(),
            record,
            estimates.advantage,
            n_epochs=n_epochs,
            n_minibatches_per_epoch=n_minibatches_per_epoch,
        ))
        self.critic_trainer(input=record.observation, target=estimates.returns)
        return record, Aux(mean_cost=jnp.mean(record.cost))

    return members(apply)


def ppo_iteration(
    policy: Node,
    value: Node,
    plant: BaseNode,
    *,
    value_loss: LossFn | BaseNode,
    clip: float,
    entropy_weight: float,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    discount: float,
    trace: float,
    n_worlds: int,
    n_steps_per_chunk: int,
    n_epochs: int,
    n_minibatches_per_epoch: int,
    n_chunks_per_minibatch: int,
    n_critic_passes: int,
) -> Node:
    """Assembles one continuing-control PPO iteration from `policy`, `value`, and `plant`."""
    n_chunks_per_epoch = n_minibatches_per_epoch * n_chunks_per_minibatch
    n_rows_per_minibatch = n_worlds * n_chunks_per_minibatch
    observation = initial_observation(plant)
    policy = policy.with_input(observation)
    value = value.with_input(observation)

    # Inner scan carries plant and policy state within each chunk; outer scanned
    # carries them across chunks within an iteration and resets between iterations.
    sampler = externalize(
        batch(
            scanned(
                scan(SamplingStep(policy, plant), n=n_steps_per_chunk),
                n=n_chunks_per_epoch,
            ),
            n=n_worlds,
        ),
        'policy',
    )
    replay = batch(ChunkReplay(scan(CommandLikelihood(policy))), n=n_rows_per_minibatch)
    trajectory_value = batch(
        batch(batch(value, n=n_steps_per_chunk), n=n_chunks_per_epoch),
        n=n_worlds,
    )
    trajectory_advantage = StandardizedAdvantageEstimates(batch(
        advantage_estimates(discount=discount, trace=trace),
        n=n_worlds,
    ))
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
    return PPOIteration(
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
    """Parameterizes and executes a PPO `training_program`, returning the trained policy, final iteration, and loss history."""
    iteration, aux = split_aux(
        jax.jit(
            training_program.parameterize(rng=parameter_key).apply,
        )(rng=training_key),
    )
    return Struct(
        policy=iteration.actor_trainer.trained().pnode.policy,
        iteration=iteration,
        history=Struct(
            actor_loss=aux.training.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss.reshape(-1),
            approximate_kl=aux.training.actor_trainer.objective.loss.approximate_kl.reshape(-1),
            clip_fraction=aux.training.actor_trainer.objective.loss.clip_fraction.reshape(-1),
            entropy=aux.training.actor_trainer.objective.loss.entropy.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
