"""Soft Actor-Critic over a replay Buffer that is ordinary Node state.

The buffer's currency is one chunk per row: an observation sequence with
one step of overlap, the commands and costs taken along it, and the policy
state at the chunk start. A feed-forward policy leaves the stored state
empty and unread, while a recurrent policy resumes from it, so one code
path serves both lifecycles with no fork anywhere.

One iteration collects fresh chunks through a sampler whose policy
parameters arrive as data, inserts them, draws every minibatch of the
iteration in one gather, and scans one gradient update over them. One
gradient update holds the actor, critic, and temperature trainers, the
target critic, and the EMA that feeds it. The actor trainer differentiates
a replay that values its fresh commands with the online critic, whose
parameters arrive as data; the critic fits the soft Bellman target formed
from that same replay one step later under the target critic; the
temperature steps on the replay's log-probabilities. Actor, critic, and
temperature update from the pre-update snapshot of each other's
parameters, which the call order expresses. The assembly builds all of it
from a policy, a critic, and a plant with stock transforms.
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
    PyTree,
    Struct,
    Wrapper,
    batch,
    externalize,
    node,
    scan,
    scanned,
    split_aux,
    tile,
    tree_reshape,
    tree_tail,
    train_step,
)
from nodejax import nn
from examples.rl.control import SamplingStep
from examples.rl.replay import Buffer


@node
def SampledCommand(policy: Node) -> Node:
    """Replay one stored observation, drawing a fresh command."""
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

    Applying it reads the temperature back; ``value`` is the same read as a
    method, so a holder of trained weights never touches the stored form.
    """
    def param():
        return jnp.log(jnp.asarray(initial))

    def value(param) -> jax.Array:
        return jnp.exp(param)

    return Leaf(value, param=param, methods={'value': value})


@node
def temperature_loss(target_entropy: float) -> Node:
    """Configure the temperature objective from sampled log-probabilities.

    The gradient with respect to the stored logarithm raises the temperature
    while the policy's entropy sits below the target and lowers it above.
    ``logprob`` is shaped (chunk, time + 1); both axes are averaged.
    """
    def apply(temperature: jax.Array, logprob: jax.Array) -> jax.Array:
        return -jnp.log(temperature) * (jnp.mean(logprob) + target_entropy)

    return Leaf(apply)


def entropy_regularized_cost(output: Struct, temperature: jax.Array) -> tuple[jax.Array, Aux]:
    """SAC's actor objective: critic cost plus entropy pressure.

    ``output`` is a valued replay, fresh commands with their
    log-probabilities and critic costs shaped (chunk, time + 1), while
    ``temperature`` is scalar. Both replay axes are averaged. Only the
    commands carry gradient into the cost, and that path is the actor's
    signal. The policy's mean entropy estimate rides on aux.
    """
    loss = jnp.mean(output.cost + temperature * output.logprob)
    return loss, Aux(entropy=-jnp.mean(output.logprob))


@node
def ValuedReplay(replay: Node, critic: Node) -> Node:
    """Replay stored observations with fresh commands and cost each one.

    Observation leaves begin (chunk, time + 1, ...), and ``initial`` is one
    policy-state pytree per chunk whose leaves begin (chunk, ...). ``replay``
    is the sampling step under ``scan``, batched over chunks, so its state is
    the policy's per chunk; bound to ``initial`` at the call, it resumes the
    memory observed during collection and produces fresh commands and
    log-probabilities on the observation axes. ``critic`` maps observation
    and command on those same axes to a cost. Externalizing ``critic`` lets
    a caller supply the online critic's parameters on every call and keeps
    them out of this Node's own parameter tree.
    """
    members = Composite(replay=replay, critic=critic)

    def apply(self, observation, initial):
        replayed = self.replay.bind(state=initial)(observation=observation)
        cost = self.critic(Struct(
            observation=replayed.observation,
            command=replayed.command,
        ))
        return replayed.replace(cost=cost)

    return members(apply)


@node
def SACUpdate(
    actor_trainer: Node,
    target_critic: Node,
    critic_trainer: Node,
    temperature_trainer: Node,
    ema_critic: Node,
    *,
    discount: float,  # per-step discount on the soft Bellman target
) -> Node:
    """One SAC gradient update over one replayed minibatch of chunks.

    Observation leaves begin (chunk, time + 1, ...), command leaves begin
    (chunk, time, ...), ``cost`` is shaped (chunk, time), and ``initial`` is one
    recorded policy-state pytree per chunk whose leaves begin (chunk, ...).

    ``actor_trainer`` differentiates a valued replay under an externalized
    ``critic`` and returns the fresh commands and log-probabilities it
    stepped through. ``target_critic`` costs the replayed tail, its whole
    parameter tree arriving in its ``critic`` field. ``critic_trainer``
    fits the critic to the soft Bellman target and ``temperature_trainer``
    steps the temperature on the replay's log-probabilities; both expose
    what they hold as ``params()``. ``ema_critic`` smooths the critic's
    parameters into the target critic. The output is the replay of the
    pre-update forward pass; the temperature rides on aux.
    """
    members = Composite(
        actor_trainer=actor_trainer,
        target_critic=target_critic,
        critic_trainer=critic_trainer,
        temperature_trainer=temperature_trainer,
        ema_critic=ema_critic,
    )

    def apply(self, observation, command, cost, initial):
        alpha = self.temperature_trainer.trained().value()
        critic_param = self.critic_trainer.params()
        ema_critic_param = self.ema_critic(critic_param)
        replayed = self.actor_trainer(
            observation=observation,
            initial=initial,
            critic=critic_param,
            temperature=alpha,
        )
        # The Bellman target reads the same replay one step later.
        tail = tree_tail(Struct(
            observation=observation,
            command=replayed.command,
            logprob=replayed.logprob,
        ), axis=1)
        future = self.target_critic(
            input=Struct(
                observation=tail.observation,
                command=tail.command,
            ),
            critic=ema_critic_param,
        )
        targets = cost + discount * (future + alpha * tail.logprob)
        head = jax.tree.map(lambda value: value[:, :-1], observation)
        self.critic_trainer(input=Struct(observation=head, command=command), target=targets)
        self.temperature_trainer(logprob=replayed.logprob)
        return replayed, Aux(temperature=alpha)

    def init(param):
        """Start the target critic at the critic's initial weights, so the
        update initializes without a minibatch."""
        critic_state = critic_trainer.bind(param.critic_trainer).init()
        fitted = critic_trainer.bind(param.critic_trainer, state=critic_state)
        return Struct(
            actor_trainer=actor_trainer.bind(param.actor_trainer).init(),
            critic_trainer=critic_state,
            temperature_trainer=temperature_trainer.bind(
                param.temperature_trainer).init(),
            ema_critic=ema_critic.bind(()).init(input=fitted.params()),
        )

    return members(apply, init=init)


@node
def SACIteration(
    sampler: Node,
    buffer: Node,
    update: Node,
    *,
    n_updates: int,  # gradient updates per iteration, the update's scan length
    n_chunks_per_minibatch: int,  # buffer rows one gradient update consumes
) -> Node:
    """One off-policy SAC iteration: collect, store, and update from replay.

    Every call starts a fresh sampled rollout from ``initial_plant_state``
    leaves shaped (world, ...). ``disturbance`` is shaped (world, chunk, time),
    and sampled-record leaves begin (world, chunk, time, ...). The (world,
    chunk) axes are flattened into the replay buffer's row axis while time
    remains within each row.

    ``sampler`` rolls the policy out over chunks and worlds, its policy
    parameters arriving in its ``policy`` field, and records the
    observation, the policy state each step saw, the command, the cost, and
    the next observation. ``buffer`` stores one chunk per row and draws
    minibatches through ``sample``. ``update`` is one gradient update
    scanned over the iteration's minibatches; it holds the policy the
    sampler runs, read through its actor trainer. The output is the
    collected rollout; the mean cost per step rides on aux.
    """
    members = Composite(sampler=sampler, buffer=buffer, update=update)

    def apply(self, initial_plant_state, disturbance, rng):
        record = self.sampler(
            disturbance=disturbance,
            initial_plant_state=initial_plant_state,
            policy=self.update.actor_trainer.params().replay,
        )

        # One overlap step lets a single replay pass serve both the fresh commands and the
        # shifted Bellman targets. The buffer's currency is one chunk per row: flattening the
        # (world, chunk) axes decorrelates sampling across both.
        last = jax.tree.map(lambda value: value[:, :, -1:], record.next_observation)
        sequence = jax.tree.map(
            lambda head, end: jnp.concatenate((head, end), axis=2),
            record.observation,
            last,
        )
        chunks = Struct(observation=sequence, command=record.command, cost=record.cost)
        rows = tree_reshape(chunks, (-1,), axes=2)
        chunk_starts = tree_reshape(
            jax.tree.map(lambda value: value[:, :, 0], record.policy_state),
            (-1,),
            axes=2,
        )
        self.buffer(rows.replace(initial=chunk_starts))

        # One gather supplies every minibatch of the iteration; all draws see the buffer with
        # this iteration's chunks inserted.
        drawn = self.buffer.sample(n_updates * n_chunks_per_minibatch, rng=rng.next())
        self.update(bundle=tree_reshape(drawn, (n_updates, n_chunks_per_minibatch)))
        return record, Aux(mean_cost=jnp.mean(record.cost))

    return members(apply)


def sac_iteration(
    policy: Node,
    critic: Node,
    plant: Node,
    *,
    critic_loss: LossFn | BaseNode,  # fits critic output to the soft Bellman target
    transition: PyTree,  # one zero transition: observation, command, cost of a step
    capacity: int,  # replay rows held
    discount: float,  # per-step discount on the soft Bellman target
    actor_optimizer,  # Optax transformation for the policy parameters
    critic_optimizer,  # Optax transformation for the critic parameters
    temperature_optimizer,  # Optax transformation for the temperature
    ema_critic_decay: float,  # EMA decay of the target critic's parameters
    target_entropy: float,  # entropy the temperature steers the policy toward
    initial_temperature: float,
    n_worlds: int,
    n_chunks: int,  # chunks collected per world per iteration
    n_steps_per_chunk: int,  # chunk length, also the replay's truncation depth
    n_updates: int,  # gradient updates per iteration
    n_chunks_per_minibatch: int,  # buffer rows one gradient update consumes
) -> Node:
    """Assemble one SAC iteration from a policy, a critic, and a plant.

    ``critic_loss`` fits the state-command cost Node to a target and may
    declare ``aux``. The transition resolves the policy and critic contracts
    and shapes the buffer's rows together with the policy's own state.
    """
    policy = policy.with_input(transition.observation)
    critic = critic.with_input(Struct(
        observation=transition.observation,
        command=transition.command,
    ))
    element = Struct(
        observation=tile(transition.observation, n_steps_per_chunk + 1),
        command=tile(transition.command, n_steps_per_chunk),
        cost=tile(transition.cost, n_steps_per_chunk),
        initial=policy.state_spec,
    )
    sampler = externalize(
        batch(
            scanned(scan(SamplingStep(policy, plant), n=n_steps_per_chunk)),
            n=n_worlds,
        ),
        'policy',
    )
    replay = batch(scan(SampledCommand(policy)), n=n_chunks_per_minibatch)
    minibatch_critic = batch(
        batch(critic, n=n_steps_per_chunk),
        n=n_chunks_per_minibatch,
    )
    sequence_critic = batch(
        batch(critic, n=n_steps_per_chunk + 1),
        n=n_chunks_per_minibatch,
    )
    update = SACUpdate(
        actor_trainer=train_step(
            externalize(ValuedReplay(replay, sequence_critic), 'critic'),
            entropy_regularized_cost,
            actor_optimizer,
        ),
        target_critic=externalize(minibatch_critic, field='critic'),
        critic_trainer=train_step(minibatch_critic, critic_loss, critic_optimizer),
        temperature_trainer=train_step(
            Temperature(initial_temperature),
            temperature_loss(target_entropy),
            temperature_optimizer,
        ),
        ema_critic=nn.EMA(tau=ema_critic_decay, warm=True),
        discount=discount,
    )
    return SACIteration(
        sampler=sampler,
        buffer=Buffer(capacity, element),
        update=scan(update, n=n_updates),
        n_updates=n_updates,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
    )


def sac_training(
    program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run a SAC program and return what it trained.

    ``program`` carries a SAC iteration over its data. The result holds the
    trained policy, read out of the actor trainer's trained model, the
    iteration bound to its final state, from which a caller binds a critic
    to the fitted parameters, and the history.
    """
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    return Struct(
        policy=final.update.actor_trainer.trained().pnode.replay.policy,
        iteration=final,
        history=Struct(
            actor_loss=aux.training.update.actor_trainer.loss.reshape(-1),
            critic_loss=aux.training.update.critic_trainer.loss.reshape(-1),
            temperature=aux.training.update.temperature[..., -1],
            entropy=aux.training.update.actor_trainer.objective.loss.entropy.reshape(-1),
            mean_cost=aux.training.mean_cost,
        ),
    )
