"""Recurrent PPO on the pendulum: the hidden-state buffer is just data.

Recurrent PPO is the version everybody hates writing: on-policy
trajectories are replayed for a few epochs under moving parameters,
and a recurrent policy must resume every replayed chunk from the
memory it actually had during collection. Here the rollout scan
records its own state trajectory as auxiliary output, a chunk's
stored state is ordinary data sliced from it, and replay is a fresh
scan primed from that data. The first test pins the property exactly:
replaying a chunk under the collection parameters reproduces the
rollout's log-probabilities.

The same learner takes a feed-forward or a GRU policy, the on-policy
counterpart of the SHAC example's lifecycle point. The plant is the
shared weak pendulum; the policy is Gaussian with a learned
state-independent spread; advantages are generalized advantage
estimates from a separate value network on privileged state.

Beyond PPO itself, the example demonstrates where responsibilities
live. The plant owns physics and observability, the policy owns its
distribution, a step owns one transition's wiring, and the learner
owns optimization. Measured by what is ABSENT next to the same
algorithm in raw JAX:

- no anonymous carry tuples, and no hand-threading of them
- no towers of nested function transforms
- no bare vmaps with axis bookkeeping: batching is a named transform
  around a step that does not know it is batched, and minibatches are
  a nested batch over the (chunks, worlds) axes
- no manual rng splitting or threading
- no done-flag arithmetic: episode boundaries are declared, never
  multiplied into carries
- no stop_gradient scattering: the gradient boundary sits where the
  trainer sits
- no TrainState, no (params, opt_state, apply_fn) triples in function
  signatures: the trainer's carry is one value
- no replay-buffer surgery: the rollout's outputs are the buffer, and
  the recurrent-state trajectory is one record=True
- no jit ceremony: two plain jits at the loop boundary, no static or
  donated argument bookkeeping
- no dummy-batch initialization dance: one observation spec, once
- no duplicated evaluation path: the deterministic step differs from
  the sampling step by one line

What remains is statements about pendulums, policies, and PPO, and
nothing that is a statement about JAX. The usual alternatives hand
you the networks and leave the rest as raw JAX around them; a
side-by-side against a stock implementation (brax ships a PPO) is the
natural comparison to write some day.

The suite trains briefly and checks improvement; the full swing-up
lives in main(): ``python -m examples.rl.test_ppo_pendulum``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (
    BaseNode,
    Composite,
    Leaf,
    Node,
    Struct,
    batch,
    node,
    scan,
    split_aux,
    tile,
    train_step,
)
from nodejax import nn
from examples.rl.pendulum import Pendulum, PendulumFeatures

HIDDEN = 64
MEMORY = 32
WORLDS = 32
HORIZON = 128
CHUNK = 16
EPOCHS = 4
MINIBATCHES = 4
CLIP = 0.2
DISCOUNT = 0.97
TRACE = 0.95
ENTROPY_WEIGHT = 1e-3
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
CRITIC_PASSES = 4


@node
def Spread() -> Node:
    """A learned state-independent log standard deviation."""
    def param():
        return jnp.asarray(-0.5)

    def apply(param):
        return param

    return Leaf(apply, param=param)


@node
def GaussianPolicy(memory: BaseNode = nn.identity) -> Node:
    """Observation to action-distribution parameters; acyclic with the
    identity memory and cyclic with a GRU, same public contract."""
    members = Composite(
        features=PendulumFeatures(),
        encoder=nn.Linear(HIDDEN) >> nn.silu,
        memory=memory,
        mean=(
            nn.Linear(HIDDEN)
            >> nn.silu
            >> nn.Projection(weight_init=jax.nn.initializers.zeros)
        ),
        spread=Spread(),
    )

    def apply(self, input):
        encoded = self.encoder(self.features(input))
        representation = self.memory(encoded)
        return Struct(mean=self.mean(representation),
                      log_std=self.spread())

    return members(apply)


@node
def Value() -> Node:
    """Privileged state value for the advantage baseline."""
    members = Composite(
        features=PendulumFeatures(),
        body=(
            nn.Linear(HIDDEN)
            >> nn.tanh
            >> nn.Linear(HIDDEN)
            >> nn.tanh
            >> nn.Projection()
        ),
    )

    def apply(self, input):
        return self.body(self.features(input))

    return members(apply)


def gaussian_logprob(command: jax.Array, proposal: Struct) -> jax.Array:
    spread = jnp.exp(proposal.log_std)
    return (
        -0.5 * ((command - proposal.mean) / spread) ** 2
        - proposal.log_std
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )


@node
def SamplingStep(policy: Node, plant: Node) -> Node:
    """One on-policy transition: sample, act, record what replay needs."""
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state, rng):
        observation = self.plant.observe()
        proposal = self.policy(observation)
        command = proposal.mean + jnp.exp(proposal.log_std) * (
            jax.random.normal(rng.next(), proposal.mean.shape))
        output = self.plant(command=command, disturbance=disturbance)
        return Struct(
            observation=observation,
            command=command,
            logprob=gaussian_logprob(command, proposal),
            cost=output.cost,
            next_observation=output.state,
        )

    def init(self, input):
        """Start a rollout from the physical state its caller supplies;
        a recurrent policy's memory initializes from its observation."""
        if not policy.cyclic:
            return Struct(plant=input.initial_state)
        observation = plant.observe(state=input.initial_state)
        return Struct(
            policy=policy.bind(self.policy).init(input=observation),
            plant=input.initial_state,
        )

    return members(apply, init=init)


@node
def ReplayStep(policy: Node) -> Node:
    """Re-evaluate one stored transition under the current parameters.

    The recurrent state at the chunk start arrives as data: prime
    adopts ``initial`` verbatim, so replay resumes exactly the memory
    the rollout had. The field rides every step of the chunk and is
    read once."""
    members = Composite(policy=policy)

    def apply(self, observation, command, initial):
        proposal = self.policy(observation)
        entropy = proposal.log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        return Struct(
            logprob=gaussian_logprob(command, proposal),
            entropy=entropy,
        )

    if not policy.cyclic:
        # no memory to restore: the initial field rides along unread
        return members(apply)

    def init(self, input):
        return input.initial

    return members(apply, init=init)


def ppo_loss(output: Struct, target: Struct) -> jax.Array:
    """Clipped surrogate plus an entropy bonus, cost-flavored: costs
    shrink, so the surrogate maximizes negative-cost advantage."""
    ratio = jnp.exp(output.logprob - target.logprob)
    clipped = jnp.clip(ratio, 1.0 - CLIP, 1.0 + CLIP)
    surrogate = jnp.minimum(ratio * target.advantage,
                            clipped * target.advantage)
    return -(jnp.mean(surrogate) + ENTROPY_WEIGHT * jnp.mean(output.entropy))


def mse(output: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((output - target) ** 2)


def advantage_estimates(reward: jax.Array, value: jax.Array,
                        final_value: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over a time-major rollout."""
    next_value = jnp.concatenate([value[1:], final_value[None]])
    delta = reward + DISCOUNT * next_value - value

    def step(carry, input):
        estimate = input + DISCOUNT * TRACE * carry
        return estimate, estimate

    advantage = jax.lax.scan(
        step, jnp.zeros_like(final_value), delta, reverse=True)[1]
    return advantage, advantage + value


def random_starts(rng: np.random.RandomState, worlds: int) -> Struct:
    return Struct(
        angle=jnp.asarray(rng.uniform(-np.pi, np.pi, worlds)),
        velocity=jnp.asarray(rng.uniform(-2.0, 2.0, worlds)),
    )


def collect(sampler: Node, policy_param, initial_state: Struct,
            key: jax.Array) -> tuple[Struct, Struct]:
    """One on-policy rollout; returns the per-step record and the
    recorded state trajectory the replay chunks start from."""
    disturbance = jnp.zeros((HORIZON, WORLDS))
    starts = tile(initial_state, HORIZON)
    rollout = sampler.bind(Struct(policy=policy_param)).initialize(
        input=Struct(disturbance=disturbance, initial_state=starts))
    _, output = rollout.apply(
        disturbance=disturbance, initial_state=starts, rng=key)
    record, aux = split_aux(output)
    return record, aux.state


def chunked(tree, count: int, length: int):
    """(horizon, worlds, ...) -> (length, count, worlds, ...): the time
    axis split into chunks, laid out time-major for the replay scan,
    with the (chunks, worlds) axes kept whole for the nested batch.
    The one axis move of the pipeline lives here."""
    return jax.tree.map(
        lambda value: jnp.swapaxes(
            value.reshape(count, length, *value.shape[1:]), 0, 1),
        tree)


@node
def MeanStep(policy: Node, plant: Node) -> Node:
    """One deterministic transition for evaluation: the mean action."""
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state):
        observation = self.plant.observe()
        proposal = self.policy(observation)
        output = self.plant(command=proposal.mean, disturbance=disturbance)
        return Struct(cost=output.cost, state=output.state,
                      action=output.action)

    def init(self, input):
        if not policy.cyclic:
            return Struct(plant=input.initial_state)
        observation = plant.observe(state=input.initial_state)
        return Struct(
            policy=policy.bind(self.policy).init(input=observation),
            plant=input.initial_state,
        )

    return members(apply, init=init)


def ppo_training(policy: Node, iterations: int) -> tuple[Struct, list]:
    """Iterate collect, estimate, replay-update; returns the trained
    policy parameters and the per-iteration mean costs."""
    plant = Pendulum()
    sampler = scan(batch(SamplingStep(policy, plant), n=WORLDS),
                   record=True)
    count = HORIZON // CHUNK
    minibatch_chunks = count // MINIBATCHES
    observation = Struct(angle=jnp.zeros(()), velocity=jnp.zeros(()))

    policy_weights = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0)).param
    value = Value().with_input(observation).parameterize(
        rng=jax.random.PRNGKey(1))
    trajectory_value = batch(batch(Value(), n=WORLDS), n=HORIZON,
                             axis='time')
    terminal_value = batch(Value(), n=WORLDS)

    # nested batching over (chunks, worlds); a recurrent replay
    # resumes a carried state over the chunk's time axis, a
    # feed-forward one is a pure map over it
    wide = batch(batch(ReplayStep(policy), n=WORLDS), n=minibatch_chunks)
    replay = (scan(wide) if policy.cyclic
              else batch(wide, n=CHUNK, axis='time'))
    actor = train_step(replay.bind(Struct(policy=policy_weights)),
                       ppo_loss, optax.adam(ACTOR_RATE))
    critic = train_step(trajectory_value.bind(value.param), mse,
                        optax.adam(CRITIC_RATE))
    advance = jax.jit(lambda carry, bundle, target: actor.apply(
        carry, input=bundle, target=target))
    critic_advance = jax.jit(lambda carry, observation, target: critic.apply(
        carry, input=observation, target=target))

    shuffle = np.random.RandomState(2)
    actor_carry = None
    critic_carry = critic.initialize().state
    history = []
    for iteration in range(iterations):
        weights = (actor_carry.opt.params.policy
                   if actor_carry is not None else policy_weights)
        record, states = collect(
            sampler, weights, random_starts(shuffle, WORLDS),
            jax.random.PRNGKey(100 + iteration))
        history.append(float(jnp.mean(record.cost)))

        critic_weights = critic_carry.opt.params
        value_trajectory = trajectory_value.bind(critic_weights).apply(
            record.observation)
        final_value = terminal_value.bind(critic_weights).apply(
            jax.tree.map(lambda value: value[-1], record.next_observation))
        advantage, returns = advantage_estimates(
            -record.cost, value_trajectory, final_value)
        advantage = (advantage - jnp.mean(advantage)) / (
            jnp.std(advantage) + 1e-8)

        rows = chunked(Struct(
            observation=record.observation,
            command=record.command,
            logprob=record.logprob,
            advantage=advantage,
        ), count, CHUNK)
        # the recurrent state each chunk started from: time-major rows
        # put the chunk's first step at rows[0], and slicing the state
        # trajectory the same way selects exactly those steps
        starts = jax.tree.map(
            lambda value: value[::CHUNK], states.policy)

        for epoch in range(EPOCHS):
            order = shuffle.permutation(count)
            for piece in range(MINIBATCHES):
                chosen = order[piece * minibatch_chunks:
                               (piece + 1) * minibatch_chunks]
                bundle = Struct(
                    observation=jax.tree.map(
                        lambda value: value[:, chosen],
                        rows.observation),
                    command=rows.command[:, chosen],
                    initial=tile(Struct(policy=jax.tree.map(
                        lambda value: value[chosen], starts)), CHUNK),
                )
                target = Struct(
                    logprob=rows.logprob[:, chosen],
                    advantage=rows.advantage[:, chosen],
                )
                if actor_carry is None:
                    actor_carry = actor.initialize(input=Struct(
                        input=bundle, target=target)).state
                actor_carry, _ = advance(actor_carry, bundle, target)

        for fitting in range(CRITIC_PASSES):
            critic_carry, _ = critic_advance(
                critic_carry, record.observation, returns)
    return (actor_carry.opt.params.policy
            if actor_carry is not None else policy_weights), history


def mean_rollout(policy: Node, weights, starts: Struct,
                 steps: int) -> Struct:
    """Deterministic closed-loop rollout from the given starts."""
    worlds = starts.angle.shape[0]
    disturbance = jnp.zeros((steps, worlds))
    initial = tile(starts, steps)
    rollout = scan(batch(MeanStep(policy, Pendulum()), n=worlds)).bind(
        Struct(policy=weights)).initialize(
        input=Struct(disturbance=disturbance, initial_state=initial))
    return split_aux(rollout.apply(
        disturbance=disturbance, initial_state=initial)[1])[0]


def evaluation_cost(policy: Node, weights, steps: int = 300) -> Struct:
    """Deterministic rollouts from hanging starts."""
    from examples.rl.pendulum import downward_starts
    trajectory = mean_rollout(policy, weights, downward_starts(), steps)
    final = jax.tree.map(lambda value: value[-1], trajectory.state)
    return Struct(
        mean_cost=jnp.mean(trajectory.cost),
        final_angle=jnp.max(jnp.abs(final.angle)),
        final_velocity=jnp.max(jnp.abs(final.velocity)),
    )


def test_ppo_improves_both_policy_lifecycles() -> None:
    """A short run of the full machinery moves the rollout cost for a
    feed-forward and a recurrent policy alike. PPO earns its cost drop
    from samples alone, so twenty iterations buy about a tenth: the
    contrast with the SHAC example's differentiable shortcut is the
    point. The swing-up itself is main()'s budget."""
    for policy in (GaussianPolicy(), GaussianPolicy(nn.GRU(MEMORY))):
        weights, history = ppo_training(policy, iterations=20)
        assert np.all(np.isfinite(history))
        assert np.mean(history[-4:]) < 0.95 * np.mean(history[:4]), history


def test_replay_reproduces_the_rollout() -> None:
    """The stored-state property, pinned: replaying every chunk from
    its recorded state under the collection parameters reproduces the
    rollout's log-probabilities."""
    policy = GaussianPolicy(nn.GRU(MEMORY))
    plant = Pendulum()
    sampler = scan(batch(SamplingStep(policy, plant), n=WORLDS),
                   record=True)
    observation = Struct(angle=jnp.zeros(()), velocity=jnp.zeros(()))
    weights = policy.with_input(observation).parameterize(
        rng=jax.random.PRNGKey(0)).param

    record, states = collect(
        sampler, weights, random_starts(np.random.RandomState(0), WORLDS),
        jax.random.PRNGKey(1))
    count = HORIZON // CHUNK
    rows = chunked(Struct(observation=record.observation,
                          command=record.command,
                          logprob=record.logprob), count, CHUNK)
    starts = jax.tree.map(lambda value: value[::CHUNK], states.policy)

    replay = scan(batch(batch(ReplayStep(policy), n=WORLDS), n=count))
    initial = tile(Struct(policy=starts), CHUNK)
    replayed = replay.bind(Struct(policy=weights)).initialize(
        input=Struct(observation=rows.observation, command=rows.command,
                     initial=initial),
    ).apply(observation=rows.observation, command=rows.command,
            initial=initial)[1]

    assert jnp.allclose(split_aux(replayed)[0].logprob, rows.logprob,
                        atol=1e-5)


def swing_up() -> None:
    """The full training run and its phase portrait, main() only."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.rl.pendulum import (
        downward_starts, phase_grid, phase_portrait, overlay_trajectories)

    policy = GaussianPolicy(nn.GRU(MEMORY))
    weights, history = ppo_training(policy, iterations=400)
    result = evaluation_cost(policy, weights)
    print(f'training cost {history[0]:.3f} -> {history[-1]:.3f} | '
          f'final angle {result.final_angle:.3f} rad | '
          f'final velocity {result.final_velocity:.3f} rad/s')

    grid = phase_grid()
    flat = jax.tree.map(lambda value: value.reshape(-1), grid)
    slice_trajectory = mean_rollout(policy, weights, flat, steps=1)
    actions = slice_trajectory.action[0].reshape(grid.angle.shape)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    phase_portrait(axis, grid, actions)
    rollouts = mean_rollout(policy, weights, downward_starts(), steps=300)
    overlay_trajectories(axis, rollouts.state)
    axis.set_title('recurrent PPO pendulum: fresh-state policy slice, '
                   'closed-loop rollouts')
    output = os.path.join(os.path.dirname(__file__), 'plots',
                          'ppo_pendulum_phase_space.png')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    swing_up()
