"""Pendulum policy, value, data, and evaluation Nodes for PPO."""

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    node,
    scanned,
    tile,
    tree_last,
    tree_len,
)
from nodejax import nn
from examples.rl.control import ControlledStep
from examples.rl.pendulum import (
    PendulumFeatures,
    VELOCITY_SCALE,
)


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
def PendulumValue(hidden: int) -> Node:
    """Pendulum state value for the advantage baseline."""
    return (
        PendulumFeatures()
        >> nn.Linear(hidden)
        >> nn.tanh
        >> nn.Linear(hidden)
        >> nn.tanh
        >> nn.Projection()
    )


@node
def PendulumTrainingData(
    iterations: int,
    *,
    n_worlds: int,
    n_chunks: int,
    n_steps_per_chunk: int,
) -> PNode:
    """Independent starts and disturbances for one collection-driven run."""
    def apply(rng):
        sample = jax.random.uniform(rng.next(), (2, iterations, n_worlds))
        return Struct(
            initial_state=Struct(
                angle=2.0 * jnp.pi * sample[0] - jnp.pi,
                velocity=VELOCITY_SCALE * (2.0 * sample[1] - 1.0),
            ),
            disturbance=jnp.zeros(
                (iterations, n_chunks, n_steps_per_chunk, n_worlds),
            ),
        )

    return Leaf(apply)


@node
def ProposalMean() -> Node:
    """Select the deterministic command from a Gaussian proposal."""
    return Leaf(lambda input: input.mean)


def mean_rollout_program(
    policy: PNode,
    plant: BaseNode,
    n_worlds: int,
) -> PNode:
    """Build a fresh deterministic rollout from an injected policy and plant."""
    mean_policy = policy >> ProposalMean()
    return scanned(
        batch(ControlledStep(mean_policy, plant), n=n_worlds)
    ).parameterize()


def mean_rollout(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    disturbance: jax.Array,
) -> Struct:
    """Run a deterministic closed loop from the given starts."""
    n_worlds = tree_len(starts)
    steps = tree_len(disturbance)
    input = Struct(
        disturbance=disturbance,
        initial_state=tile(starts, steps),
    )
    rollout = mean_rollout_program(policy, plant, n_worlds)
    trajectory = rollout.apply(bundle=input)
    return Struct(
        cost=trajectory.cost,
        state=trajectory.next_state,
        action=trajectory.action,
    )


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
        jnp.zeros((steps, tree_len(starts))),
    )
    final = tree_last(trajectory.state)
    return Struct(
        mean_cost=jnp.mean(trajectory.cost),
        final_angle=jnp.max(jnp.abs(final.angle)),
        final_velocity=jnp.max(jnp.abs(final.velocity)),
    )
