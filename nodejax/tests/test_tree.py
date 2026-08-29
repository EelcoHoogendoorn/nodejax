"""Generic pytree operations."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import (
    Struct, tree_broadcast_axis, tree_first, tree_last, tree_len,
    tree_stop_gradient,
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


def test_tree_len_and_broadcast_axis():
    tree = Struct(
        left=jnp.zeros((3, 2)),
        right=jnp.ones((3, 4, 1)),
    )
    assert tree_len(tree) == 3

    repeated = tree_broadcast_axis(tree, 5, axis=1)
    assert repeated.left.shape == (3, 5, 2)
    assert repeated.right.shape == (3, 5, 4, 1)
    assert jnp.array_equal(repeated.left[:, 0], tree.left)


def test_tree_len_rejects_no_common_leading_axis():
    with pytest.raises(ValueError, match='at least one leaf'):
        tree_len(())
    with pytest.raises(ValueError, match='leading axis'):
        tree_len(Struct(value=jnp.asarray(1.0)))
    with pytest.raises(ValueError, match='unequal'):
        tree_len(Struct(left=jnp.zeros(2), right=jnp.zeros(3)))


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
