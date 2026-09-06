"""What the learners share: the policy's body, the critic's view of the
plant, the starts they train and evaluate from, the evaluation of a
trained policy, and its artifacts.

The policy sees only what the actuator's own controller knows, the
observer's angle and velocity estimate and the estimated quadrature
current, and the target; it sees the rope only when the plant has a tip
sensor and the policy's features read it. The critic, which is
training-time only, sees the rope.
"""

import math
import time
from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import Leaf, Node, PNode, Struct, node, serial
from nodejax import nn
from examples.rl import control
from examples.whip.plots import animate_rollout, plot_actuator_readouts
from examples.whip.whip import N_LINKS, PLANT_DT


HIDDEN = 64
MEMORY = 32
# The critic maps forty-odd rope coordinates and a target to a value; at 64 it explained three
# quarters of its targets' variance and stopped, at 256 it kept climbing past 85 percent.
CRITIC_HIDDEN = 256
VELOCITY_SCALE = 20.0  # rad/s: the unit the policy reads velocities in and writes setpoints in
CURRENT_SCALE = 50.0
SPEED_SCALE = 20.0
# A chunk's summed hits, in units of the hit energy, run into the tens; the critic's head
# works in units of this.
VALUE_SCALE = 10.0
N_EVALUATION_STEPS = 150
EVALUATION_ANGLES = (0.0, 0.5 * math.pi, math.pi)  # along the floor, up, and along it the other way
START_ANGLES = (0.1, math.pi - 0.1)  # training starts, above the floor
# Targets are drawn in polar coordinates about the pivot: a whip crack lands
# at some fraction of full extension along some direction, so this covers
# the crackable spots and none inside the arm's shadow or past the tip.
TARGET_RADIUS = (1.0, 1.6)  # of a 1.8 m reach
TARGET_ANGLES = (math.pi / 6.0, 5.0 * math.pi / 6.0)  # the upper arc, clear of the floor
# (radius, angle) pairs
EVALUATION_TARGETS = ((1.2, 0.5 * math.pi), (1.5, 0.25 * math.pi), (1.0, 0.75 * math.pi))


def polar_target(radius: float | jax.Array, angle: float | jax.Array) -> jax.Array:
    """A target from its radius and angle about the pivot."""
    return jnp.stack((radius * jnp.cos(angle), radius * jnp.sin(angle)), axis=-1)


def random_targets(rng: jax.Array, shape: tuple) -> jax.Array:
    """Targets drawn uniformly in radius and angle over the training ranges,
    shaped ``shape + (2,)``."""
    radius_key, angle_key = jax.random.split(rng)
    radius = jax.random.uniform(radius_key, shape, minval=TARGET_RADIUS[0], maxval=TARGET_RADIUS[1])
    angle = jax.random.uniform(angle_key, shape, minval=TARGET_ANGLES[0], maxval=TARGET_ANGLES[1])
    return polar_target(radius, angle)


def random_starts(plant: PNode, rng: jax.Array, shape: tuple) -> Struct:
    """Starts of ``plant`` for training, shaped ``shape``: its rope at rest
    at a random angle above the floor, and a random target."""
    angle_key, target_key = jax.random.split(rng)
    angles = jax.random.uniform(angle_key, shape, minval=START_ANGLES[0], maxval=START_ANGLES[1])
    return plant.start(angle=angles, target=random_targets(target_key, shape))


def evaluation_starts(plant: PNode) -> Struct:
    """The fixed starts a policy on ``plant`` is evaluated from: every
    evaluation angle with every evaluation target, one world per pair."""
    angles, targets = [], []
    for angle in EVALUATION_ANGLES:
        for radius, direction in EVALUATION_TARGETS:
            angles.append(angle)
            targets.append(polar_target(radius, direction))
    return plant.start(angle=jnp.array(angles), target=jnp.stack(targets))


@node
def WhipFeatures(tip_sensor: bool) -> Node:
    """The controllers' estimates, per joint, and the target, as one
    feature vector; with ``tip_sensor`` also where the rope's end is and
    how fast it moves, for the privileged policy on a plant that senses
    its tip."""
    def apply(input):
        fields = [
            jnp.cos(input.angle),
            jnp.sin(input.angle),
            input.velocity / VELOCITY_SCALE,
            input.current / CURRENT_SCALE,
            input.target,
        ]
        if tip_sensor:
            fields += [input.tip_position, input.tip_velocity / SPEED_SCALE]
        return jnp.concatenate(fields, axis=-1)

    return Leaf(apply)


@node
def WhipMean(features: Node, memory: Node, hidden: int) -> Node:
    """Whip observation to the mean of one velocity setpoint per joint, in
    rad/s, with memory; ``features`` is what the policy reads of the
    observation. The head works in the unit the features read velocities
    in and the last stage is that unit; nothing bounds the setpoint, the
    actuator clips what it cannot follow."""
    return serial(
        features=features,
        encoder=nn.Linear(hidden) >> nn.silu,
        memory=memory,
        command=(
            nn.Linear(hidden)
            >> nn.silu
            >> nn.Linear(N_LINKS, weight_init=jax.nn.initializers.zeros)
        ),
        unit=Leaf(lambda input: input * VELOCITY_SCALE),
    )


@node
def RopeFeatures() -> Node:
    """What the critic sees of the plant state: the rope in world
    coordinates and the target, as a flat vector."""
    def apply(input):
        rope = input.rope
        return jnp.concatenate((
            rope.position.reshape(-1),
            rope.velocity.reshape(-1) / SPEED_SCALE,
            input.target,
        ))

    return Leaf(apply)


@node
def RelativeRopeFeatures() -> Node:
    """The critic's view with the rope's positions taken from the target,
    so the geometry the value depends on arrives as such, and the target
    itself beside it for the floor and the arm's reach."""
    def apply(input):
        rope = input.rope
        return jnp.concatenate((
            (rope.position - input.target).reshape(-1),
            rope.velocity.reshape(-1) / SPEED_SCALE,
            input.target,
        ))

    return Leaf(apply)


@node
def WhipCritic(features: Node, hidden: int, value_scale: float) -> Node:
    """A scalar value of the plant state from ``features`` of its rope, in
    units of ``value_scale``: the targets are summed tip energies in the
    tens, and a head initialized near zero would take an age to grow
    there."""
    return serial(
        features=features,
        body=(
            nn.Linear(hidden)
            >> nn.tanh
            >> nn.Linear(hidden)
            >> nn.tanh
            >> nn.Projection()
        ),
        scale=Leaf(lambda input: input * value_scale),
    )


def whip_critic(plant: PNode, features: Node = RopeFeatures(), hidden: int = CRITIC_HIDDEN) -> Node:
    """The critic of ``plant`` reading ``features`` of its state, its input
    shape read from the plant at rest."""
    return WhipCritic(features, hidden, VALUE_SCALE).with_input(control.initial_state(plant))


def untrained_policy(plant: PNode, features: Node, rng: jax.Array) -> PNode:
    """A fresh policy at the plant's observation, the baseline."""
    return WhipMean(features, nn.GRU(MEMORY), HIDDEN).with_input(
        control.initial_observation(plant)).parameterize(rng=rng)


def whip_evaluation(
    policy: PNode,
    plant: PNode,
    starts: Struct,
    rng: jax.Array,
    steps: int = N_EVALUATION_STEPS,
    step: Callable = control.ControlledStep,
) -> Struct:
    """Closed-loop rollouts of a mean policy on ``plant`` from ``starts``,
    shaped (world,): the state trajectory, the joints read off it, the
    policy's commands, the torque, and the reward each world collected, the
    negative cost summed over time. ``step`` is ``ControlledStep`` for a
    policy on observations, ``PlannedStep`` for a planner on states."""
    trajectory = control.policy_trajectory(policy, plant, starts, steps, rng, step=step)
    return Struct(
        state=trajectory.state,
        mechanical=plant.mechanical(state=trajectory.state),
        command=trajectory.command,
        action=trajectory.action,
        reward=-jnp.sum(trajectory.cost, axis=1),
    )


def timed(train: Callable) -> Struct:
    """A training run and the wall seconds it took, compilation included;
    ``train`` returns a Struct with a ``history`` whose ``mean_cost`` is the
    run's last value to arrive."""
    started = time.perf_counter()
    result = train()
    jax.block_until_ready(result.history.mean_cost)
    return Struct(result=result, seconds=time.perf_counter() - started)


def closest_approach(evaluation: Struct) -> Struct:
    """Per world, how near the tip came to the target and when, at the
    policy's clock."""
    target = evaluation.state.target[:, :1]  # constant over time; keep the axis for broadcasting
    distance = jnp.linalg.norm(evaluation.state.rope.position[:, :, -1] - target, axis=-1)
    return Struct(
        distance=jnp.min(distance, axis=1),
        time=jnp.argmin(distance, axis=1) * PLANT_DT,
    )


def summarize(evaluation: Struct, label: str) -> None:
    """Print the rewards and closest approaches of an evaluation, per world
    from its start angle and target."""
    approach = closest_approach(evaluation)
    angles = evaluation.mechanical.position[:, 0, 0]
    targets = evaluation.state.target[:, 0]
    print(f'{label}:')
    for world in range(angles.shape[0]):
        target = targets[world]
        print(
            f'  start {float(angles[world]):+.2f} rad, '
            f'target ({float(target[0]):+.2f}, {float(target[1]):+.2f}): '
            f'reward {float(evaluation.reward[world]):8.1f}, '
            f'closest {float(approach.distance[world]):.2f} m '
            f'at {float(approach.time[world]):.2f} s'
        )
    print(f'  mean reward {float(jnp.mean(evaluation.reward)):.1f}, '
          f'mean closest approach {float(jnp.mean(approach.distance)):.2f} m')
    print(f'  correlation between the joints\' commands over all worlds '
          f'{float(command_correlation(evaluation)):.2f}')


def command_correlation(evaluation: Struct) -> jax.Array:
    """How alike the two joints' commands are: the correlation coefficient
    of the shoulder's and the elbow's command over every world and step."""
    shoulder = evaluation.command[..., 0].reshape(-1)
    elbow = evaluation.command[..., 1].reshape(-1)
    return jnp.corrcoef(shoulder, elbow)[0, 1]


def write_animation(evaluation: Struct, prefix: str, world: int = 0) -> None:
    """Write the animation of one world."""
    animation = animate_rollout(
        evaluation.state.rope.position[world], evaluation.state.target[world, 0],
        f'{prefix}_policy.gif')
    print(f'animation: {animation}')


def write_artifacts(evaluation: Struct, prefix: str, world: int = 0) -> None:
    """Write the animation and the actuator readouts of one world."""
    write_animation(evaluation, prefix, world)
    readouts = plot_actuator_readouts(evaluation, world, f'{prefix}_actuator_readouts.png')
    print(f'readouts: {readouts}')
