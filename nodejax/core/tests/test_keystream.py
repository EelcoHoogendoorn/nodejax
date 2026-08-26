"""Narrow authored RNG streams and compositional transform capabilities."""

import jax
import jax.numpy as jnp
import pytest

from nodejax.core.rng import KeyStream, MaybeKeyStream


def assert_key_equal(left, right):
    assert jnp.array_equal(jax.random.key_data(left),
                           jax.random.key_data(right))


def test_authored_stream_draws_in_a_reproducible_order():
    key = jax.random.key(3)
    left = KeyStream(key)
    right = KeyStream(key)

    assert_key_equal(left.next(), right.next())
    assert_key_equal(left.next(), right.next())


def test_authored_stream_has_no_transform_routing_surface():
    rng = KeyStream(jax.random.key(4))

    with pytest.raises(AttributeError):
        rng.child
    with pytest.raises(AttributeError):
        rng.axis
    with pytest.raises(AttributeError):
        rng.split
    with pytest.raises(AttributeError):
        rng.broadcast


def test_empty_capability_has_no_key_to_draw():
    rng = MaybeKeyStream()

    assert not rng
    with pytest.raises(TypeError, match='random key required'):
        rng.next()


def test_deterministic_children_do_not_advance_the_parent():
    key = jax.random.key(4)
    actual = MaybeKeyStream(key)
    expected = KeyStream(key)

    assert not actual.child(False)
    assert_key_equal(actual.next(), expected.next())


def test_stochastic_children_receive_distinct_streams():
    rng = MaybeKeyStream(jax.random.key(5))
    left = rng.child(True)
    right = rng.child(True)

    assert left and right
    assert not jnp.array_equal(
        jax.random.key_data(left.next()),
        jax.random.key_data(right.next()),
    )


def test_axis_returns_split_or_broadcast_capabilities():
    count = 3
    random_rng, random_axis = MaybeKeyStream(jax.random.key(6)).axis(
        True, count)
    empty_rng, empty_axis = MaybeKeyStream().axis(False, count)

    assert random_axis == 0
    assert empty_axis is None
    assert not empty_rng
    draws = jax.vmap(
        lambda rng: jax.random.normal(rng.next()),
        in_axes=random_axis,
    )(random_rng)
    assert draws.shape == (count,)


def test_compositional_capabilities_are_jax_pytrees():
    keyed = MaybeKeyStream(jax.random.key(7))
    empty = MaybeKeyStream()

    keyed_leaves, keyed_tree = jax.tree.flatten(keyed)
    empty_leaves, empty_tree = jax.tree.flatten(empty)
    rebuilt = jax.tree.unflatten(keyed_tree, keyed_leaves)

    assert len(keyed_leaves) == 1
    assert empty_leaves == []
    assert_key_equal(rebuilt.next(), KeyStream(jax.random.key(7)).next())
    assert not jax.tree.unflatten(empty_tree, empty_leaves)


def test_capability_rejects_invalid_axis_requests():
    rng = MaybeKeyStream(jax.random.key(8))

    with pytest.raises(TypeError, match='takes_rng must be a bool'):
        rng.child(1)
    with pytest.raises(TypeError, match='non-negative int'):
        rng.split(-1)
    with pytest.raises(TypeError, match='known count'):
        rng.axis(True, None)
