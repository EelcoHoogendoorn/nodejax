"""The weakly actuated pendulum shared by the control examples.

Upright is angle zero. The actuator saturates well below gravity, so
no policy can lift the mass directly: swinging up means pumping energy
across many steps, which is what makes this tiny plant a real test of
credit over time. The plant is an ordinary stateful Node: transitions
differentiate, the physical state has named fields, and ``observe`` is
a node method giving a controller's view of the state.
"""

import math

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    node,
    tree_last,
    tree_len,
)
from nodejax import nn
from examples.rl.control import mean_rollout


DT = 0.1
MAX_TORQUE = 0.5
ACTION_COST = 0.01
VELOCITY_COST = 0.1
VELOCITY_SCALE = 2.0


@node
def PendulumFeatures() -> Node:
    """Periodic coordinates and signed energy error for any controller."""

    def apply(input):
        cosine = jnp.cos(input.angle)
        energy_error = 0.5 * input.velocity**2 + cosine - 1.0
        return jnp.stack(
            (
                cosine,
                jnp.sin(input.angle),
                input.velocity / VELOCITY_SCALE,
                energy_error,
                energy_error * input.velocity / VELOCITY_SCALE**2,
            ),
            axis=-1,
        )

    return Leaf(apply)


@node
def PendulumMean(
    memory: BaseNode,
    hidden: int,
    features: BaseNode | None = None,
) -> Node:
    """Pendulum observation to Gaussian mean, optionally with memory.

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
        return self.command(representation)

    return members(apply)


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
def AngleFeatures() -> Node:
    """Periodic coordinates of the angle alone.

    For a controller denied the velocity: consecutive angles are then the
    only route to it, so this is the honest workload for a policy with
    memory. Consumers of the full observation ignore the velocity field.
    """
    def apply(input):
        return jnp.stack(
            (jnp.cos(input.angle), jnp.sin(input.angle)),
            axis=-1,
        )

    return Leaf(apply)


@node
def Pendulum(
    dt: float = DT,
    max_torque: float = MAX_TORQUE,
    static_friction: float = 0.0,
) -> Node:
    """Weak actuator, upright angle zero, and named physical state fields."""
    if not math.isfinite(static_friction) or static_friction < 0.0:
        raise ValueError('static friction must be finite and non-negative')

    def init():
        return Struct(
            angle=jnp.zeros(()),
            velocity=jnp.zeros(()),
        )

    def apply(state, command, disturbance):
        action = max_torque * jnp.tanh(command)
        drive = jnp.sin(state.angle) + action + disturbance
        if static_friction == 0.0:
            velocity = state.velocity + dt * drive
        else:
            free_velocity = state.velocity + dt * drive
            moving = jnp.abs(state.velocity) > 1e-6
            crosses_zero = moving & (state.velocity * free_velocity <= 0.0)
            held = (~moving) & (jnp.abs(drive) <= static_friction)
            velocity = jnp.where(held | crosses_zero, 0.0, free_velocity)
        angle = state.angle + dt * velocity
        angle = jnp.arctan2(jnp.sin(angle), jnp.cos(angle))
        next_state = Struct(angle=angle, velocity=velocity)
        cost = (
            2.0 * (1.0 - jnp.cos(state.angle))
            + VELOCITY_COST * state.velocity**2
            + ACTION_COST * action**2
        )
        return next_state, Struct(
            state=next_state,
            action=action,
            cost=cost,
        )

    def observe(state) -> Struct:
        """What a controller may see: for this plant, the full state."""
        return state

    return Leaf(apply, init=init, methods={'observe': observe})


@node
def PendulumTrainingData(
    iterations: int,
    *,
    n_worlds: int,
    n_chunks: int,
    n_steps_per_chunk: int,
) -> PNode:
    """Independent starts and disturbances for one collection-driven run.

    Initial-state leaves are shaped (iteration, world, ...), and ``disturbance``
    is shaped (iteration, world, chunk, time).
    """
    def apply(rng):
        sample = jax.random.uniform(rng.next(), (2, iterations, n_worlds))
        return Struct(
            initial_state=Struct(
                angle=2.0 * jnp.pi * sample[0] - jnp.pi,
                velocity=VELOCITY_SCALE * (2.0 * sample[1] - 1.0),
            ),
            disturbance=jnp.zeros(
                (iterations, n_worlds, n_chunks, n_steps_per_chunk),
            ),
        )

    return Leaf(apply)


def pendulum_evaluation(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    steps: int,
) -> Struct:
    """Summarize a deterministic Pendulum rollout from injected starts."""
    trajectory = mean_rollout(
        policy,
        plant,
        starts,
        jnp.zeros((tree_len(starts), steps)),
    )
    final = tree_last(trajectory.state, axis=1)
    return Struct(
        mean_cost=jnp.mean(trajectory.cost),
        final_angle=jnp.max(jnp.abs(final.angle)),
        final_velocity=jnp.max(jnp.abs(final.velocity)),
    )


PHASE_VELOCITY_LIMIT = 3.5
ENERGY_CONTOUR_COLOR = '#c7ad7f'
RANDOM_TRAJECTORY_COLOR = '#293746'


def phase_grid(resolution: int = 61) -> Struct:
    """A meshed phase plane over one angle wrap and the plotting
    velocity range, as state fields ready for a batched policy."""
    import numpy as np
    angles, velocities = np.meshgrid(
        np.linspace(-np.pi, np.pi, resolution),
        np.linspace(-PHASE_VELOCITY_LIMIT, PHASE_VELOCITY_LIMIT, resolution),
    )
    return Struct(angle=jnp.asarray(angles), velocity=jnp.asarray(velocities))


def phase_portrait(axis, grid: Struct) -> None:
    """Draw the common analytic background for real phase trajectories."""
    import numpy as np

    angles = np.asarray(grid.angle)
    velocities = np.asarray(grid.velocity)
    energy = 0.5 * velocities**2 + np.cos(angles)

    axis.contour(
        angles, velocities, energy,
        levels=(-0.5, 0.0, 0.5, 1.0, 1.5),
        colors=ENERGY_CONTOUR_COLOR, linewidths=0.75,
    )
    axis.plot(
        (),
        (),
        color=ENERGY_CONTOUR_COLOR,
        linewidth=0.9,
        label='passive energy contours',
    )

    axis.scatter(0.0, 0.0, marker='*', color='black', s=130, zorder=4,
                 label='upright equilibrium')
    axis.scatter((-np.pi, np.pi), (0.0, 0.0), marker='x', color='black',
                 s=50, label='downward equilibrium')
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(-PHASE_VELOCITY_LIMIT, PHASE_VELOCITY_LIMIT)
    axis.set_xlabel(r'angle $\theta$ [rad]')
    axis.set_ylabel(r'angular velocity $\omega$ [rad/s]')
    axis.grid(alpha=0.14)


def phase_surface(
    axis,
    grid: Struct,
    values: jax.Array,
    label: str,
) -> None:
    """Draw a scalar surface over the pendulum phase plane."""
    import numpy as np

    angles = np.asarray(grid.angle)
    velocities = np.asarray(grid.velocity)
    values = np.asarray(values)
    lower = np.floor(values.min())
    upper = np.ceil(values.max())
    if upper <= lower:
        upper = lower + 1.0
    levels = np.linspace(lower, upper, 21)
    surface = axis.contourf(
        angles,
        velocities,
        values,
        levels=levels,
        cmap='magma',
        extend='both',
    )
    axis.contour(
        angles,
        velocities,
        values,
        levels=levels[::2],
        colors='white',
        linewidths=0.45,
        alpha=0.5,
    )
    axis.figure.colorbar(surface, ax=axis, label=label)
    axis.scatter(0.0, 0.0, marker='*', color='white', s=80, zorder=4)
    axis.scatter(
        (-np.pi, np.pi),
        (0.0, 0.0),
        marker='x',
        color='white',
        s=35,
        zorder=4,
    )
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(-PHASE_VELOCITY_LIMIT, PHASE_VELOCITY_LIMIT)
    axis.set_xlabel(r'angle $\theta$ [rad]')
    axis.grid(alpha=0.14)


def overlay_trajectories(
    axis,
    state: Struct,
    colors=None,
    *,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    mark_starts: bool = True,
) -> None:
    """Overlay closed-loop rollouts shaped (world, time), splitting at the
    angle wrap."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    states = np.stack(
        (np.asarray(state.angle), np.asarray(state.velocity)),
        axis=-1,
    )
    worlds = states.shape[0]
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, worlds))
    for index in range(worlds):
        points = states[index]
        segments = [
            (points[step], points[step + 1])
            for step in range(points.shape[0] - 1)
            if abs(points[step + 1, 0] - points[step, 0]) < np.pi
        ]
        axis.add_collection(LineCollection(
            segments,
            colors=[colors[index]],
            linewidths=linewidth,
            alpha=alpha,
        ))
    if mark_starts:
        axis.scatter(
            states[:, 0, 0],
            states[:, 0, 1],
            color=colors,
            s=38,
            edgecolor='white',
            linewidth=0.8,
            alpha=alpha,
            zorder=3,
        )


def overlay_phase_trajectories(axis, state: Struct) -> None:
    """Draw colored downward references and thin sampled trajectories."""
    import matplotlib.pyplot as plt
    import numpy as np

    references = downward_starts().angle.shape[0]
    reference_state = jax.tree.map(
        lambda value: value[:references],
        state,
    )
    random_state = jax.tree.map(
        lambda value: value[references:],
        state,
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, references))
    overlay_trajectories(axis, reference_state, colors)
    overlay_trajectories(
        axis,
        random_state,
        (RANDOM_TRAJECTORY_COLOR,) * random_state.angle.shape[0],
        linewidth=0.75,
        alpha=0.52,
        mark_starts=False,
    )
    if random_state.angle.shape[0]:
        axis.scatter(
            np.asarray(random_state.angle[:, 0]),
            np.asarray(random_state.velocity[:, 0]),
            facecolor='none',
            edgecolor=RANDOM_TRAJECTORY_COLOR,
            s=17,
            linewidth=0.7,
            zorder=3,
            label='random rollout start',
        )


def downward_starts() -> Struct:
    """Hanging starts near the stable equilibrium, the hard side."""
    angle = jnp.asarray((
        -jnp.pi,
        -jnp.pi + 0.05,
        -jnp.pi + 0.1,
        jnp.pi,
        jnp.pi - 0.05,
        jnp.pi - 0.1,
    ))
    return Struct(angle=angle, velocity=jnp.zeros_like(angle))


def phase_starts(
    key: jax.Array,
    random_trajectories: int = 24,
    velocity_limit: float = PHASE_VELOCITY_LIMIT,
) -> Struct:
    """Downward references followed by reproducible random phase states."""
    angle_key, velocity_key = jax.random.split(key)
    random = Struct(
        angle=jax.random.uniform(
            angle_key,
            (random_trajectories,),
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        velocity=jax.random.uniform(
            velocity_key,
            (random_trajectories,),
            minval=-velocity_limit,
            maxval=velocity_limit,
        ),
    )
    return jax.tree.map(
        lambda reference, sampled: jnp.concatenate((reference, sampled)),
        downward_starts(),
        random,
    )
