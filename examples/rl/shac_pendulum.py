"""Pendulum policy, critic, data, evaluation, and plotting support for SHAC."""

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    drop_aux,
    node,
    tile,
    tree_first,
    tree_last,
    tree_len,
)
from nodejax import nn
from examples.rl.control import ControlledStep
from examples.rl.pendulum import (
    MAX_TORQUE,
    VELOCITY_SCALE,
    PendulumFeatures,
    downward_starts,
    overlay_trajectories,
    phase_grid,
    phase_portrait,
    phase_surface,
)


@node
def PendulumPolicy(
    memory: BaseNode,
    hidden: int,
    features: BaseNode | None = None,
) -> Node:
    """A deterministic policy with an explicit memory lifecycle.

    ``features`` narrows what the policy sees; the default reads the full
    observation."""
    members = Composite(
        features=features or PendulumFeatures(),
        encoder=nn.Linear(hidden) >> nn.silu,
        memory=memory,
        command=(
            nn.Linear(hidden)
            >> nn.silu
            >> nn.Projection(weight_init=jax.nn.initializers.zeros)
        ),
    )

    def apply(self, input):
        encoded = self.encoder(self.features(input))
        representation = self.memory(encoded)
        command = self.command(representation)
        return command, Aux(representation=representation)

    return members(apply)


@node
def ScalarMLP(hidden: int) -> Node:
    """A replaceable scalar function approximator."""
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
    members = Composite(
        features=PendulumFeatures(),
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
        return phase_cost + correction

    return members(apply)


@node
def PendulumTrainingData(
    *,
    n_episodes: int,
    n_chunks_per_episode: int,
    n_steps_per_chunk: int,
    n_worlds: int,
    disturbance_scale: float,
) -> PNode:
    """Independent episodes split into short gradient chunks."""
    def apply(rng):
        disturbance = disturbance_scale * jax.random.normal(
            rng.next(),
            (n_episodes, n_chunks_per_episode, n_steps_per_chunk, n_worlds),
        )
        initial = Struct(
            angle=jax.random.uniform(
                rng.next(),
                (n_episodes, n_worlds),
                minval=-jnp.pi,
                maxval=jnp.pi,
            ),
            velocity=jax.random.uniform(
                rng.next(),
                (n_episodes, n_worlds),
                minval=-VELOCITY_SCALE,
                maxval=VELOCITY_SCALE,
            ),
        )
        # Nested scans map every input field. Only episode priming consumes the
        # duplicated initial state.
        initial_state = jax.tree.map(
            lambda value: jnp.broadcast_to(
                value[:, None, None],
                (n_episodes, n_chunks_per_episode, n_steps_per_chunk) + value.shape[1:],
            ),
            initial,
        )
        return Struct(
            disturbance=disturbance,
            initial_state=initial_state,
        )

    return Leaf(apply)


def policy_trajectory(
    policy: PNode,
    plant: BaseNode,
    initial_state: Struct,
    steps: int,
) -> Struct:
    """Evaluate from fresh state while preserving recurrent carry."""
    n_worlds = tree_len(initial_state)
    input = Struct(
        disturbance=jnp.zeros((steps, n_worlds)),
        initial_state=tile(initial_state, steps),
    )
    world = batch(
        ControlledStep(drop_aux(policy), plant),
        n=n_worlds,
    ).parameterize()
    rollout = world.initialize(input=tree_first(input))
    _, trajectory = rollout.scan(bundle=input)
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


def pendulum_evaluation(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    steps: int,
) -> Struct:
    trajectory = policy_trajectory(policy, plant, starts, steps)
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
    plant: BaseNode,
) -> str:
    """Render a policy slice, closed-loop rollouts, and the EMA critic."""
    import os
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    grid_state = phase_grid()
    flat_state = jax.tree.map(lambda value: value.reshape(-1), grid_state)
    n_worlds = tree_len(flat_state)

    # A recurrent policy has no state-only flow field. This is its first
    # action from freshly initialized member states; the overlaid trajectories
    # below are the real closed-loop paths with memory carried through time.
    action = policy_trajectory(policy, plant, flat_state, steps=1).action[0]
    actions = action.reshape(grid_state.angle.shape)
    value = batch(terminal_value, n=n_worlds).apply(flat_state)
    terminal_cost = drop_aux(value).reshape(grid_state.angle.shape)

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

    references = tree_len(downward_starts())
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
