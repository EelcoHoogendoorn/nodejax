"""Pendulum Q function, replay transition, and training data for SAC."""

import jax
import jax.numpy as jnp

from nodejax import Composite, Leaf, Node, PNode, Struct, node
from nodejax import nn
from examples.rl.pendulum import PendulumFeatures, VELOCITY_SCALE


@node
def PendulumQ(hidden: int) -> Node:
    """Pendulum state-command cost-to-go for the soft Bellman backup."""
    members = Composite(
        features=PendulumFeatures(),
        body=(
            nn.Linear(hidden)
            >> nn.tanh
            >> nn.Linear(hidden)
            >> nn.tanh
            >> nn.Projection()
        ),
    )

    def apply(self, input):
        observed = self.features(input.observation)
        return self.body(jnp.append(observed, input.command))

    return members(apply)


def pendulum_transition() -> Struct:
    """One zero transition fixing the replay element's fields and shapes."""
    observation = Struct(angle=jnp.zeros(()), velocity=jnp.zeros(()))
    return Struct(
        observation=observation,
        command=jnp.zeros(()),
        cost=jnp.zeros(()),
        next_observation=observation,
    )


@node
def PendulumTrainingData(
    iterations: int,
    *,
    n_worlds: int,
    n_steps_per_world: int,
) -> PNode:
    """Independent starts and zero disturbances for one Pendulum SAC run."""
    def apply(rng):
        sample = jax.random.uniform(rng.next(), (2, iterations, n_worlds))
        return Struct(
            initial_state=Struct(
                angle=2.0 * jnp.pi * sample[0] - jnp.pi,
                velocity=VELOCITY_SCALE * (2.0 * sample[1] - 1.0),
            ),
            disturbance=jnp.zeros((iterations, n_steps_per_world, n_worlds)),
        )

    return Leaf(apply)