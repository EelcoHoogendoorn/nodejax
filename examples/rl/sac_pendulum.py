"""Pendulum Q function and replay transition for SAC."""

import jax
import jax.numpy as jnp

from nodejax import Composite, Node, Struct, node
from nodejax import nn
from examples.rl.pendulum import PendulumFeatures


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
