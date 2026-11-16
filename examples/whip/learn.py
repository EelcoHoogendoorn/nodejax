"""Provides shared policy/critic architectures, start distributions, rollout evaluation, and artifact writers for whip learners."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

from nodejax import BaseNode, Leaf, Node, PNode, Struct, node, serial
from nodejax import nn
from examples.rl import control
from examples.whip.plots import animate_rollout, plot_actuator_readouts
from examples.whip.whip import N_LINKS, PLANT_DT


HIDDEN = 64
MEMORY = 32
CRITIC_HIDDEN = 256
VELOCITY_SCALE = 20.0  # rad/s scale for policy velocity inputs and setpoints
CURRENT_SCALE = 50.0
SPEED_SCALE = 20.0
VALUE_SCALE = 10.0
N_EVALUATION_STEPS = 150
EVALUATION_ANGLES = (0.0, 0.5 * math.pi, math.pi)
START_ANGLES = (0.1, math.pi - 0.1)
TARGET_RADIUS = (1.0, 1.6)  # out of a 1.8 m total reach
TARGET_ANGLES = (math.pi / 6.0, 5.0 * math.pi / 6.0)
EVALUATION_TARGETS = ((1.2, 0.5 * math.pi), (1.5, 0.25 * math.pi), (1.0, 0.75 * math.pi))


def polar_target(radius: float | jax.Array, angle: float | jax.Array) -> jax.Array:
    """Converts polar `(radius, angle)` coordinates about the pivot into Cartesian `(x, y)` coordinates."""
    return jnp.stack((radius * jnp.cos(angle), radius * jnp.sin(angle)), axis=-1)


def random_targets(rng: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Samples 2D Cartesian targets of shape `shape + (2,)` uniformly in polar radius and angle."""
    radius_key, angle_key = jax.random.split(rng)
    radius = jax.random.uniform(radius_key, shape, minval=TARGET_RADIUS[0], maxval=TARGET_RADIUS[1])
    angle = jax.random.uniform(angle_key, shape, minval=TARGET_ANGLES[0], maxval=TARGET_ANGLES[1])
    return polar_target(radius, angle)


def random_starts(plant: PNode, rng: jax.Array, shape: tuple[int, ...]) -> Struct:
    """Samples initial `plant` states of shape `shape` with resting ropes at random angles and random targets."""
    angle_key, target_key = jax.random.split(rng)
    angles = jax.random.uniform(angle_key, shape, minval=START_ANGLES[0], maxval=START_ANGLES[1])
    return plant.start(angle=angles, target=random_targets(target_key, shape))


def evaluation_starts(plant: PNode) -> Struct:
    """Builds the deterministic evaluation grid pairing each angle in `EVALUATION_ANGLES` with `EVALUATION_TARGETS`."""
    angles, targets = [], []
    for angle in EVALUATION_ANGLES:
        for radius, direction in EVALUATION_TARGETS:
            angles.append(angle)
            targets.append(polar_target(radius, direction))
    return plant.start(angle=jnp.array(angles), target=jnp.stack(targets))


@node
def WhipFeatures(tip_sensor: bool) -> PNode:
    """Packs joint angle cosines/sines, normalized velocity and current estimates, target coordinates, and optional tip states."""
    def apply(input: Struct) -> jax.Array:
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
def WhipMean(features: BaseNode, memory: BaseNode, hidden: int) -> Node:
    """Maps whip observations through `features` and recurrent `memory` to unbounded joint velocity setpoints in `rad/s`."""
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
def RopeFeatures() -> PNode:
    """Flattens world-frame rope positions, normalized rope velocities, and target coordinates into one critic feature vector."""
    def apply(input: Struct) -> jax.Array:
        rope = input.rope
        return jnp.concatenate((
            rope.position.reshape(-1),
            rope.velocity.reshape(-1) / SPEED_SCALE,
            input.target,
        ))

    return Leaf(apply)


@node
def RelativeRopeFeatures() -> PNode:
    """Flattens target-relative rope positions, normalized rope velocities, and target coordinates for critic evaluation."""
    def apply(input: Struct) -> jax.Array:
        rope = input.rope
        return jnp.concatenate((
            (rope.position - input.target).reshape(-1),
            rope.velocity.reshape(-1) / SPEED_SCALE,
            input.target,
        ))

    return Leaf(apply)


@node
def WhipCritic(features: BaseNode, hidden: int, value_scale: float) -> Node:
    """Maps plant state `features` through a two-layer `tanh` MLP to a scalar value scaled by `value_scale`."""
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


def whip_critic(plant: PNode, features: BaseNode = RopeFeatures(), hidden: int = CRITIC_HIDDEN) -> Node:
    """Constructs a `WhipCritic` bound to the input spec of `control.initial_state(plant)`."""
    return WhipCritic(features, hidden, VALUE_SCALE).with_input(control.initial_state(plant))


def untrained_policy(plant: PNode, features: BaseNode, rng: jax.Array) -> PNode:
    """Parameterizes a fresh `GRU`-backed `WhipMean` policy on `plant` observations."""
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
    """Rolls out `policy` on `plant` across `starts` and returns state, joint mechanics, commands, torques, and total rewards."""
    trajectory = control.policy_trajectory(policy, plant, starts, steps, rng, step=step)
    return Struct(
        state=trajectory.state,
        mechanical=plant.mechanical(state=trajectory.state),
        command=trajectory.command,
        action=trajectory.action,
        reward=-jnp.sum(trajectory.cost, axis=1),
    )


def timed(train: Callable[[], Struct]) -> Struct:
    """Executes `train()`, blocks on `result.history.mean_cost`, and returns `Struct(result, seconds)`."""
    started = time.perf_counter()
    result = train()
    jax.block_until_ready(result.history.mean_cost)
    return Struct(result=result, seconds=time.perf_counter() - started)


def closest_approach(evaluation: Struct) -> Struct:
    """Computes each world's minimum tip-to-target Euclidean distance and corresponding timestamp in seconds."""
    target = evaluation.state.target[:, :1]
    distance = jnp.linalg.norm(evaluation.state.rope.position[:, :, -1] - target, axis=-1)
    return Struct(
        distance=jnp.min(distance, axis=1),
        time=jnp.argmin(distance, axis=1) * PLANT_DT,
    )


def summarize(evaluation: Struct, label: str) -> None:
    """Prints per-world and mean rewards, closest tip-to-target distances, and inter-joint command correlation."""
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
    """Computes the Pearson correlation coefficient between shoulder and elbow velocity commands across all steps."""
    shoulder = evaluation.command[..., 0].reshape(-1)
    elbow = evaluation.command[..., 1].reshape(-1)
    return jnp.corrcoef(shoulder, elbow)[0, 1]


def write_animation(evaluation: Struct, prefix: str, world: int = 0) -> None:
    """Writes a rollout GIF animation for `world` under `prefix`."""
    animation = animate_rollout(
        evaluation.state.rope.position[world], evaluation.state.target[world, 0],
        f'{prefix}_policy.gif')
    print(f'animation: {animation}')


def write_artifacts(evaluation: Struct, prefix: str, world: int = 0) -> None:
    """Writes both the rollout animation and the actuator readout plot for `world` under `prefix`."""
    write_animation(evaluation, prefix, world)
    readouts = plot_actuator_readouts(evaluation, world, f'{prefix}_actuator_readouts.png')
    print(f'readouts: {readouts}')
