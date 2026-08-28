"""A small SHAC pendulum with interchangeable feed-forward and GRU policies.

The control algorithm is deliberately ordinary. It differentiates through a
short physical rollout, bootstraps its tail with a learned value, and fits that
value to stopped TD-lambda targets. The point of the example is the lifecycle:
the same learner accepts a stateless policy or a recurrent policy, while
NodeJAX carries recurrent state across gradient chunks and resets it only at
episode boundaries.

Both policies have the same public contract. Calling one returns a scalar
command, while the post-memory representation remains diagnostic ``Aux``.
The ensemble mean drives the plant, and member commands remain available in
Aux. Both policy and critic observe the full plant state.

The program uses real Node ensembles. Policy commands are averaged before the
plant acts, while the critic mean supplies bootstrap values and the retained
member values are all fitted to the same stopped targets.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Node,
    PNode,
    Struct,
    batch,
    drop_aux,
    ensemble,
    node,
    reduce,
    scan,
    scanned,
    split_aux,
    state_reinit,
    tile,
    tree_last,
    tree_stop_gradient,
    train_step,
)
from nodejax import nn
from examples.rl.pendulum import (
    MAX_TORQUE,
    VELOCITY_SCALE,
    ControlledStep,
    Pendulum,
    PendulumFeatures,
    downward_starts,
    overlay_trajectories,
    phase_grid,
    phase_portrait,
    phase_surface,
    phase_starts,
)


DISCOUNT = 0.97
TRACE = 0.95
HIDDEN = 64
MEMORY = 64
POLICY_MEMBERS = 3
CRITIC_MEMBERS = 3
HORIZON = 20
WORLDS = 64
CHUNKS = 15
EPISODES = 40
CRITIC_UPDATES = 4
TARGET_DECAY = 0.995
ACTOR_RATE = 0.001
CRITIC_RATE = 0.001
DISTURBANCE_SCALE = 0.03
EVALUATION_STEPS = 300


@node
def PendulumPolicy(memory: BaseNode = nn.identity) -> Node:
    """The same policy is acyclic with identity and cyclic with a GRU."""
    features = PendulumFeatures()
    members = Composite(
        features=features,
        encoder=nn.Linear(HIDDEN) >> nn.silu,
        memory=memory,
        command=(
            nn.Linear(HIDDEN)
            >> nn.silu
            >> nn.Projection(
                weight_init=jax.nn.initializers.zeros,
            )
        ),
    )

    def apply(self, input):
        encoded = self.encoder(self.features(input))
        representation = self.memory(encoded)
        command = self.command(representation)
        return command, Aux(representation=representation)

    return members(apply)


@node
def ScalarMLP(hidden: int = HIDDEN) -> Node:
    """A replaceable scalar function approximator with no control semantics."""
    return (
        nn.Linear(hidden)
        >> nn.tanh
        >> nn.Linear(hidden)
        >> nn.tanh
        >> nn.Projection()
    )


@node
def PendulumCritic(residual: Node) -> Node:
    """A pendulum value prior corrected by an injected scalar model."""
    features = PendulumFeatures()
    members = Composite(
        features=features,
        residual=residual,
    )

    def apply(self, input):
        observed = self.features(input)
        origin = Struct(
            angle=jnp.zeros_like(input.angle),
            velocity=jnp.zeros_like(input.velocity),
        )
        origin_features = self.features(origin)
        phase_cost = (
            2.0 * (1.0 - jnp.cos(input.angle))
            + input.velocity**2
        )
        correction = (
            self.residual(observed)
            - self.residual(origin_features)
        )
        assert correction.shape == phase_cost.shape
        return phase_cost + correction

    return members(apply)


def td_lambda(
    cost: jax.Array,
    next_value: jax.Array,
    discount: float = DISCOUNT,
    trace: float = TRACE,
) -> jax.Array:
    """Stopped TD-lambda targets for a time-major world batch."""
    assert next_value.shape == cost.shape

    def step(carry, input):
        current_cost, current_value = input
        target = current_cost + discount * (
            (1.0 - trace) * current_value + trace * carry
        )
        return target, target

    target = jax.lax.scan(
        step,
        next_value[-1],
        (cost, next_value),
        reverse=True,
    )[1]
    return jax.lax.stop_gradient(target)


def mse(output: jax.Array, target: jax.Array) -> jax.Array:
    assert output.shape == target.shape
    return jnp.mean((output - target) ** 2)


def ensemble_mse(output, target: jax.Array) -> jax.Array:
    """Fit every value retained by ``ensemble(...) >> reduce(mean)``."""
    population = split_aux(output)[1].reduce_mean.population
    assert population.shape[:-1] == target.shape
    return jnp.mean((population - target[..., None]) ** 2)


@node
def SHAC(
    policy: Node,
    critic: Node,
    plant: PNode,
    critic_loss: Callable = mse,
    worlds: int = WORLDS,
    horizon: int = HORIZON,
    critic_updates: int = CRITIC_UPDATES,
) -> Node:
    """One short-horizon policy update and one fitted-value update.

    ``critic_loss(output, target)`` returns one scalar. It may inspect Aux,
    which lets an ensemble fit all member values while exposing a scalar mean
    to the rest of SHAC.
    """
    # Bind the observation shape now: the target EMA needs critic parameters
    # before the first critic call could otherwise establish it.
    critic = critic.with_input(plant.initialize().state)

    # Carry each controlled world through physical time. The named boundary
    # resets policy and plant state only when the enclosing episode scan fires.
    world = state_reinit(
        ControlledStep(drop_aux(policy), plant),
        boundary='episode',
    )
    window = scan(batch(world, n=worlds), n=horizon)

    # SHAC consumes the critic's clean scalar output. The fitted loss receives
    # the complete output and may also use values retained in Aux.
    critic_value = drop_aux(critic)
    terminal_value = batch(critic_value, n=worlds)
    trajectory_value = batch(terminal_value, n=horizon, axis='time')
    trajectory_population = batch(
        batch(critic, n=worlds),
        n=horizon,
        axis='time',
    )
    discounts = DISCOUNT ** jnp.arange(horizon)

    def policy_loss(output, target_param) -> jax.Array:
        trajectory = output
        fixed_target = tree_stop_gradient(target_param)
        terminal = terminal_value.bind(fixed_target).apply(
            tree_last(trajectory.next_state),
        )
        running = jnp.sum(discounts[:, None] * trajectory.cost, axis=0)
        return jnp.mean(running + DISCOUNT**horizon * terminal)

    policy_step = train_step(window, policy_loss, optax.adam(ACTOR_RATE))
    critic_step = train_step(
        trajectory_population,
        critic_loss,
        optax.adam(CRITIC_RATE),
    )
    members = Composite(
        policy=policy_step,
        critic=scan(critic_step, n=critic_updates),
        target=nn.EMA(TARGET_DECAY, warm=True),
    )

    def apply(self, disturbance, initial_state):
        target_param = self.target(self.state.critic.opt.params)
        trajectory = self.policy(
            input=Struct(
                disturbance=disturbance,
                initial_state=initial_state,
            ),
            target=target_param,
        )

        next_observation = tree_stop_gradient(trajectory.next_state)
        fixed_target = tree_stop_gradient(target_param)
        next_value = trajectory_value.bind(fixed_target).apply(next_observation)
        targets = td_lambda(
            jax.lax.stop_gradient(trajectory.cost),
            next_value,
        )

        observation = tree_stop_gradient(trajectory.state)
        critic_input = tile(observation, critic_updates)
        critic_target = tile(targets, critic_updates)
        self.critic(input=critic_input, target=critic_target)
        return Struct(
            mean_cost=jnp.mean(trajectory.cost),
        )

    return members(apply)


def shac_program(policy: Node) -> Struct:
    """Build the program and retain its policy and critic for rebinding."""
    policy = ensemble(policy, n=POLICY_MEMBERS) >> reduce(jnp.mean)
    critic = (
        ensemble(PendulumCritic(ScalarMLP()), n=CRITIC_MEMBERS)
        >> reduce(jnp.mean)
    )
    learner = SHAC(
        policy,
        critic,
        Pendulum(),
        critic_loss=ensemble_mse,
    )
    episode = scan(learner, boundary='episode', n=CHUNKS)
    return Struct(
        program=scan(episode),
        policy=policy,
        critic=critic,
    )


def training_data(key: jax.Array, episodes: int = EPISODES) -> Struct:
    """Independent episodes, each split into short gradient chunks."""
    disturbance_key, angle_key, velocity_key = jax.random.split(key, 3)
    disturbance = DISTURBANCE_SCALE * jax.random.normal(
        disturbance_key,
        (episodes, CHUNKS, HORIZON, WORLDS),
    )
    initial = Struct(
        angle=jax.random.uniform(
            angle_key,
            (episodes, WORLDS),
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        velocity=jax.random.uniform(
            velocity_key,
            (episodes, WORLDS),
            minval=-VELOCITY_SCALE,
            maxval=VELOCITY_SCALE,
        ),
    )
    # Each nested scan maps every input field, so reset state needs those axes
    # too. Only episode priming consumes the duplicated value.
    initial_state = jax.tree.map(
        lambda value: jnp.broadcast_to(
            value[:, None, None],
            (episodes, CHUNKS, HORIZON) + value.shape[1:],
        ),
        initial,
    )
    return Struct(
        disturbance=disturbance,
        initial_state=initial_state,
    )


def trained_policy(
    policy: Node,
    episodes: int = EPISODES,
) -> Struct:
    """Train and return the policy, EMA critic, and learning history."""
    data = training_data(jax.random.PRNGKey(11), episodes)
    shac = shac_program(policy)
    program = shac.program.with_input(data)
    control = program.parameterize(rng=jax.random.PRNGKey(1)).initialize(
        input=data,
    )
    control, output = jax.jit(control.apply)(bundle=data)
    metric, aux = split_aux(output)
    history = Struct(
        policy_loss=aux.policy.loss.reshape(-1),
        critic_loss=aux.critic.loss[..., -1].reshape(-1),
        mean_cost=metric.mean_cost.reshape(-1),
    )
    learned = shac.policy.bind(control.state.policy.opt.params.policy)
    terminal_value = shac.critic.bind(control.state.target)
    return Struct(
        policy=learned,
        critic=terminal_value,
        history=history,
    )


def policy_trajectory_program(policy: PNode, worlds: int) -> PNode:
    """Build a fresh rollout that preserves recurrent policy carry."""
    world = ControlledStep(
        drop_aux(policy.node),
        Pendulum(),
    ).bind(
        Struct(policy=policy.param),
    )
    rollout = scanned(
        batch(world, n=worlds),
    )
    return rollout


def policy_trajectory(
    policy: PNode,
    initial_state: Struct,
    steps: int = EVALUATION_STEPS,
) -> Struct:
    """Evaluate from fresh state while preserving recurrent carry."""
    worlds = initial_state.angle.shape[0]
    input = Struct(
        disturbance=jnp.zeros((steps, worlds)),
        initial_state=tile(initial_state, steps),
    )
    rollout = policy_trajectory_program(policy, worlds)
    trajectory = rollout.apply(bundle=input)
    final_state = tree_last(trajectory.next_state)
    state = jax.tree.map(
        lambda value, final: jnp.concatenate((value, final[None]), axis=0),
        trajectory.state,
        final_state,
    )
    return Struct(
        state=state,
        action=trajectory.action,
        cost=trajectory.cost,
        final_state=final_state,
    )


def evaluation(policy: PNode) -> Struct:
    trajectory = policy_trajectory(policy, downward_starts())
    return Struct(
        cost=jnp.mean(jnp.sum(trajectory.cost, axis=0)),
        final_angle=jnp.max(jnp.abs(trajectory.final_state.angle)),
        final_velocity=jnp.max(jnp.abs(trajectory.final_state.velocity)),
        max_torque=jnp.max(jnp.abs(trajectory.action)),
    )


def plot_phase_space(
    policy: PNode,
    terminal_value: PNode,
    trajectory: Struct,
) -> str:
    """Render a policy slice, closed-loop rollouts, and the EMA critic."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    grid_state = phase_grid()
    flat_state = jax.tree.map(lambda value: value.reshape(-1), grid_state)
    worlds = flat_state.angle.shape[0]

    # A recurrent policy has no state-only flow field. This is its first
    # action from freshly initialized member states; the overlaid trajectories
    # below are the real closed-loop paths with memory carried through time.
    action = policy_trajectory(policy, flat_state, steps=1).action[0]
    actions = action.reshape(grid_state.angle.shape)
    value = batch(terminal_value, n=worlds).apply(flat_state)
    terminal_cost = split_aux(value)[0].reshape(grid_state.angle.shape)

    figure, (axis, cost_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        sharey=True,
    )
    phase_portrait(axis, grid_state, actions)
    phase_surface(
        cost_axis,
        grid_state,
        terminal_cost,
        label='learned terminal cost',
    )

    references = downward_starts().angle.shape[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, references))
    reference_state = jax.tree.map(
        lambda value: value[:, :references],
        trajectory.state,
    )
    random_state = jax.tree.map(
        lambda value: value[:, references:],
        trajectory.state,
    )
    random_colors = ('0.12',) * random_state.angle.shape[1]
    cyan_references = ('cyan',) * references
    cyan_random = ('cyan',) * random_state.angle.shape[1]
    overlay_trajectories(axis, reference_state, colors)
    overlay_trajectories(
        axis,
        random_state,
        random_colors,
        linewidth=0.75,
        alpha=0.42,
        mark_starts=False,
    )
    overlay_trajectories(
        cost_axis,
        reference_state,
        cyan_references,
        linewidth=1.2,
        alpha=0.8,
        mark_starts=False,
    )
    overlay_trajectories(
        cost_axis,
        random_state,
        cyan_random,
        linewidth=0.65,
        alpha=0.32,
        mark_starts=False,
    )
    axis.scatter(
        np.asarray(random_state.angle[0]),
        np.asarray(random_state.velocity[0]),
        facecolor='none',
        edgecolor='0.15',
        s=17,
        linewidth=0.7,
        zorder=3,
        label='random rollout start',
    )
    figure.suptitle(
        'Ensemble SHAC pendulum control',
        y=0.98,
        fontweight='bold',
    )
    axis.set_title(
        'fresh-state policy slice; real closed-loop rollouts, '
        rf'$|\tau| \leq {MAX_TORQUE:g}$',
        color='0.35',
        fontsize=10,
        pad=8,
    )
    cost_axis.set_title(
        'EMA mean terminal cost across critic members',
        color='0.35',
        fontsize=10,
        pad=8,
    )
    axis.legend(loc='upper right', frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = os.path.join(
        os.path.dirname(__file__),
        'plots',
        'pendulum_shac_phase_space.png',
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def test_policy_cyclicity_is_selected_by_its_memory_node() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = PendulumPolicy().with_input(state).parameterize(
        rng=jax.random.PRNGKey(0),
    )
    recurrent = PendulumPolicy(nn.GRU(MEMORY)).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_command, feedforward_aux = split_aux(
        feedforward.apply(state),
    )
    recurrent, recurrent_output = recurrent.apply(state)
    recurrent_command, recurrent_aux = split_aux(recurrent_output)

    assert feedforward_command.shape == recurrent_command.shape == ()
    assert feedforward.cyclic is False
    assert feedforward_aux.representation.shape == (HIDDEN,)
    assert recurrent.cyclic is True
    assert recurrent_aux.representation.shape == (MEMORY,)


def test_policy_ensembles_compile_for_both_policy_lifecycles() -> None:
    state = Struct(angle=jnp.asarray(0.4), velocity=jnp.asarray(-0.2))
    feedforward = (
        ensemble(PendulumPolicy(), n=POLICY_MEMBERS) >> reduce(jnp.mean)
    ).with_input(state).parameterize(rng=jax.random.PRNGKey(0))
    recurrent = (
        ensemble(PendulumPolicy(nn.GRU(MEMORY)), n=POLICY_MEMBERS)
        >> reduce(jnp.mean)
    ).with_input(state).parameterize(
        rng=jax.random.PRNGKey(1),
    ).initialize(input=state)

    feedforward_output = jax.jit(feedforward.apply)(state)
    recurrent, recurrent_output = jax.jit(recurrent.apply)(state)
    feedforward_command, feedforward_aux = split_aux(feedforward_output)
    recurrent_command, recurrent_aux = split_aux(recurrent_output)

    assert feedforward_command.shape == recurrent_command.shape == ()
    assert feedforward_aux.reduce_mean.population.shape == (POLICY_MEMBERS,)
    assert recurrent_aux.reduce_mean.population.shape == (POLICY_MEMBERS,)
    assert jnp.allclose(
        feedforward_command,
        jnp.mean(feedforward_aux.reduce_mean.population),
    )
    assert jnp.allclose(
        recurrent_command,
        jnp.mean(recurrent_aux.reduce_mean.population),
    )
    recurrent_state = jax.tree.leaves(recurrent.state)
    assert recurrent_state
    assert all(
        value.shape[0] == POLICY_MEMBERS
        for value in recurrent_state
    )


def test_one_shac_update_accepts_both_policy_lifecycles() -> None:
    worlds = 2
    horizon = 4
    initial = Struct(
        angle=jnp.asarray((0.4, -0.7)),
        velocity=jnp.asarray((-0.2, 0.3)),
    )
    input = Struct(
        disturbance=jnp.zeros((horizon, worlds)),
        initial_state=tile(initial, horizon),
    )

    for policy in (PendulumPolicy(), PendulumPolicy(nn.GRU(MEMORY))):
        committee = ensemble(policy, n=POLICY_MEMBERS) >> reduce(jnp.mean)
        critic = (
            ensemble(PendulumCritic(ScalarMLP()), n=CRITIC_MEMBERS)
            >> reduce(jnp.mean)
        )
        learner = SHAC(
            committee,
            critic,
            Pendulum(),
            critic_loss=ensemble_mse,
            worlds=worlds,
            horizon=horizon,
            critic_updates=1,
        )
        control = learner.with_input(input).parameterize(
            rng=jax.random.PRNGKey(2),
        ).initialize(
            input=input,
        )
        output = jax.jit(control.apply)(bundle=input)[1]
        metric, aux = split_aux(output)

        assert jnp.isfinite(metric.mean_cost)
        assert jnp.isfinite(aux.policy.loss)
        assert jnp.isfinite(aux.critic.loss).all()


def test_recurrent_state_carries_across_chunks_and_resets_at_episode() -> None:
    observations = Struct(
        angle=jnp.linspace(-2.0, 1.0, 8),
        velocity=jnp.linspace(0.5, -0.2, 8),
    )
    first = jax.tree.map(lambda value: value[:3], observations)
    second = jax.tree.map(lambda value: value[3:], observations)
    initial_input = jax.tree.map(lambda value: value[0], observations)
    policy = PendulumPolicy(nn.GRU(MEMORY)).with_input(
        initial_input,
    ).parameterize(
        rng=jax.random.PRNGKey(3),
    )
    initial_state = policy.init(input=initial_input)

    whole, whole_output = policy.bind(state=initial_state).scan(observations)
    chunked, first_output = policy.bind(state=initial_state).scan(first)
    chunked, second_output = chunked.scan(second)
    whole_command, whole_aux = split_aux(whole_output)
    first_command, first_aux = split_aux(first_output)
    second_command, second_aux = split_aux(second_output)

    assert jnp.array_equal(
        whole_command,
        jnp.concatenate((first_command, second_command)),
    )
    assert jnp.array_equal(
        whole_aux.representation,
        jnp.concatenate((
            first_aux.representation,
            second_aux.representation,
        )),
    )
    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        whole.state,
        chunked.state,
    ))

    episode = scan(
        state_reinit(
            PendulumPolicy(nn.GRU(MEMORY)),
            boundary='episode',
        ),
        boundary='episode',
    ).with_input(observations).parameterize(
        rng=jax.random.PRNGKey(3),
    ).initialize(input=observations)
    episode, first_output = episode.apply(observations)
    episode, second_output = episode.apply(observations)
    first_command, first_aux = split_aux(first_output)
    second_command, second_aux = split_aux(second_output)

    assert jnp.array_equal(first_command, second_command)
    assert jnp.array_equal(
        first_aux.representation,
        second_aux.representation,
    )


def test_shac_swings_up_with_either_policy_lifecycle() -> None:
    feedforward = trained_policy(
        PendulumPolicy(),
    )
    recurrent = trained_policy(
        PendulumPolicy(nn.GRU(MEMORY)),
    )
    feedforward_result = evaluation(feedforward.policy)
    recurrent_result = evaluation(recurrent.policy)

    assert jnp.isfinite(feedforward.history.policy_loss).all()
    assert jnp.isfinite(feedforward.history.critic_loss).all()
    assert jnp.isfinite(recurrent.history.policy_loss).all()
    assert jnp.isfinite(recurrent.history.critic_loss).all()
    for result in (feedforward_result, recurrent_result):
        assert result.final_angle < 0.1
        assert result.final_velocity < 0.1


if __name__ == '__main__':
    feedforward = trained_policy(
        PendulumPolicy(),
    )
    recurrent = trained_policy(
        PendulumPolicy(nn.GRU(MEMORY)),
    )
    feedforward_result = evaluation(feedforward.policy)
    recurrent_result = evaluation(recurrent.policy)
    print(shac_program(PendulumPolicy(nn.GRU(MEMORY))).program.describe())
    print(
        'feed-forward final error: '
        f'{feedforward_result.final_angle:.4f} rad, '
        f'{feedforward_result.final_velocity:.4f} rad/s',
    )
    print(
        'recurrent final error: '
        f'{recurrent_result.final_angle:.4f} rad, '
        f'{recurrent_result.final_velocity:.4f} rad/s',
    )
    print(
        'mean training cost per step: '
        f'{feedforward.history.mean_cost[-1]:.4f} feed-forward, '
        f'{recurrent.history.mean_cost[-1]:.4f} recurrent',
    )
    trajectory = policy_trajectory(
        recurrent.policy,
        phase_starts(jax.random.PRNGKey(29)),
    )
    portrait = plot_phase_space(
        recurrent.policy,
        recurrent.critic,
        trajectory,
    )
    print(f'phase portrait: {portrait}')
