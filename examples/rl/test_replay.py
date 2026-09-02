"""Focused behavior checks for the replay Buffer."""

import jax
import jax.numpy as jnp

from nodejax import Composite, Node, Struct, node
from examples.rl.replay import Buffer


def test_buffer_wraps_and_samples_only_filled_rows() -> None:
    capacity = 8
    element = Struct(observation=jnp.zeros(2), cost=jnp.zeros(()))
    buffer = Buffer(capacity, element).parameterize().initialize()
    segment = Struct(
        observation=jnp.arange(10.0).reshape(5, 2),
        cost=jnp.arange(1.0, 6.0),
    )
    buffer, fill = buffer.apply(segment)
    drawn = buffer.sample(64, rng=jax.random.PRNGKey(0))

    assert fill == 5
    assert jnp.all(drawn.cost >= 1.0)
    assert jnp.all(drawn.cost <= 5.0)

    buffer, fill = buffer.apply(jax.tree.map(lambda value: -value, segment))
    wrapped = jnp.array([5, 6, 7, 0, 1])

    assert fill == capacity
    assert jnp.array_equal(buffer.state.store.cost[wrapped], -segment.cost)


@node
def InsertThenSample(buffer: Node, count: int) -> Node:
    """Insert a segment and draw from the same Buffer in one apply."""
    members = Composite(buffer=buffer)

    def apply(self, segment, rng):
        fill = self.buffer(segment)
        drawn = self.buffer.sample(count, rng=rng.next())
        return Struct(fill=fill, drawn=drawn)

    return members(apply)


def test_buffer_insert_is_visible_to_a_sample_in_the_same_apply() -> None:
    element = Struct(cost=jnp.zeros(()))
    trip = InsertThenSample(
        Buffer(8, element),
        count=16,
    ).parameterize().initialize()
    segment = Struct(cost=jnp.arange(1.0, 6.0))
    trip, output = jax.jit(trip.apply)(segment, rng=jax.random.PRNGKey(1))

    assert output.fill == 5
    assert jnp.all(output.drawn.cost >= 1.0)
