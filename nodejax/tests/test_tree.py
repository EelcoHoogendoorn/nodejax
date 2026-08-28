"""Generic pytree operations."""

import jax
import jax.numpy as jnp

from nodejax import (
    Struct, tree_first, tree_last, tree_stop_gradient,
)


def test_tree_first_and_last_index_every_leaf():
    tree = Struct(
        left=jnp.asarray(((1.0, 2.0), (3.0, 4.0))),
        nested=Struct(right=jnp.asarray((5.0, 6.0))),
    )

    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        tree_first(tree),
        Struct(left=jnp.asarray((1.0, 2.0)), nested=Struct(right=5.0)),
    ))
    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        tree_last(tree),
        Struct(left=jnp.asarray((3.0, 4.0)), nested=Struct(right=6.0)),
    ))


def test_tree_stop_gradient_preserves_values_and_stops_every_leaf():
    tree = Struct(
        left=jnp.asarray((1.0, 2.0)),
        nested=Struct(right=jnp.asarray(3.0)),
    )
    stopped = tree_stop_gradient(tree)

    assert jax.tree.structure(stopped) == jax.tree.structure(tree)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, stopped, tree))

    def total(scale):
        scaled = jax.tree.map(lambda value: scale * value, tree)
        fixed = tree_stop_gradient(scaled)
        return sum(jnp.sum(value) for value in jax.tree.leaves(fixed))

    assert jax.grad(total)(jnp.asarray(1.0)) == 0.0
