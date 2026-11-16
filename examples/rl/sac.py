"""Implements Soft Actor-Critic over a cyclic replay `Buffer` carried as ordinary Node state.

Stores one chunk per replay row containing an observation sequence with one step
of overlap, the commands and costs along it, and the initial policy state. A
feed-forward policy leaves the stored policy state empty while a recurrent policy
resumes from it, allowing one replay path to serve both lifecycles.

Each `SACIteration` collects fresh chunks under an externalized policy, writes
them into `Buffer`, gathers all minibatches for the iteration, and scans
`SACUpdate` across them to step the actor, critic, entropy temperature, and EMA
target critic.
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
    PyTree,
    Struct,
    Wrapper,
    batch,
    externalize,
    node,
    scan,
    scanned,
    split_aux,
    tree_reshape,
    tree_tail,
    train_step,
)
from nodejax import nn
from examples.rl.control import SamplingStep
from examples.rl.replay import Buffer


@node
def SampledCommand(policy: BaseNode) -> Node | PNode:
    """Evaluates `policy` on one stored `observation` and samples a fresh command and log-probability."""
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
    """Stores a learned entropy temperature as its logarithm and returns its exponentiated value."""
    def param():
        return jnp.log(jnp.asarray(initial))

    def value(param) -> jax.Array:
        return jnp.exp(param)

    return Leaf(value, param=param, methods={'value': value})


@node
def temperature_loss(target_entropy: float) -> PNode:
    """Computes the dual entropy-temperature objective from `(chunk, time + 1)` log-probabilities toward `target_entropy`."""
    def apply(temperature: jax.Array, logprob: jax.Array) -> jax.Array:
        return -jnp.log(temperature) * (jnp.mean(logprob) + target_entropy)

    return Leaf(apply)


def entropy_regularized_cost(output: Struct, temperature: jax.Array) -> tuple[jax.Array, Aux]:
    """Computes SAC's actor objective averaging critic `output.cost` plus `temperature * output.logprob`."""
    loss = jnp.mean(output.cost + temperature * output.logprob)
    return loss, Aux(entropy=-jnp.mean(output.logprob))


@node
def ValuedReplay(replay: BaseNode, critic: BaseNode) -> Node | PNode:
    """Replays `(chunk, time + 1, ...)` observations from `initial` policy states and scores fresh commands with `critic`."""
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
    actor_trainer: BaseNode,
    target_critic: BaseNode,
    critic_trainer: BaseNode,
    temperature_trainer: BaseNode,
    ema_critic: BaseNode,
    *,
    discount: float,
) -> Node:
    """Executes one SAC gradient update over a replayed minibatch of chunks.

    Differentiates `actor_trainer` through `ValuedReplay` under the online critic,
    forms soft Bellman targets one step ahead under `target_critic` and `ema_critic`,
    and steps `critic_trainer` and `temperature_trainer`.
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
        # Bellman target reads the same replay shifted forward by one step.
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
        """Initializes `ema_critic` directly from `critic_trainer`'s initial parameters without requiring a minibatch."""
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
    sampler: BaseNode,
    buffer: BaseNode,
    update: BaseNode,
    *,
    n_updates: int,
    n_chunks_per_minibatch: int,
) -> Node:
    """Collects one chunked rollout across worlds, writes chunks to `buffer`, and scans `update` over sampled minibatches."""
    members = Composite(sampler=sampler, buffer=buffer, update=update)

    def apply(self, initial_plant_state, disturbance, rng):
        record = self.sampler(
            disturbance=disturbance,
            initial_plant_state=initial_plant_state,
            policy=self.update.actor_trainer.params().replay,
        )

        # Appending one overlap observation lets a single replay pass supply both
        # current-step actor commands and next-step Bellman targets.
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

        # Gather all minibatches for this iteration after inserting the new chunks.
        drawn = self.buffer.sample(n_updates * n_chunks_per_minibatch, rng=rng.next())
        self.update(bundle=tree_reshape(drawn, (n_updates, n_chunks_per_minibatch)))
        return record, Aux(mean_cost=jnp.mean(record.cost))

    return members(apply)


def sac_iteration(
    policy: Node,
    critic: Node,
    plant: BaseNode,
    *,
    critic_loss: LossFn | BaseNode,
    transition: PyTree,
    capacity: int,
    discount: float,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    temperature_optimizer: optax.GradientTransformation,
    ema_critic_decay: float,
    target_entropy: float,
    initial_temperature: float,
    n_worlds: int,
    n_chunks: int,
    n_steps_per_chunk: int,
    n_updates: int,
    n_chunks_per_minibatch: int,
) -> Node:
    """Assembles one off-policy SAC iteration from `policy`, `critic`, and `plant`."""
    policy = policy.with_input(transition.observation)
    critic = critic.with_input(Struct(
        observation=transition.observation,
        command=transition.command,
    ))
    sampler = externalize(
        batch(
            scanned(scan(SamplingStep(policy, plant), n=n_steps_per_chunk), n=n_chunks),
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
        buffer=Buffer(capacity),
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
    """Parameterizes and executes a SAC `program`, returning the trained policy, final iteration, and training history."""
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
