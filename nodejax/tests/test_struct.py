"""Verifies immutable value updates, structural merges, and pickling on `Struct`."""

import pickle

import jax.numpy as jnp
import pytest

from nodejax.struct import Struct


class SemanticStruct(Struct):
    """Represents a `Struct` subclass whose runtime type is preserved across updates."""


def test_functional_updates_preserve_semantic_subtypes() -> None:
    original = SemanticStruct(
        value=1,
        nested=SemanticStruct(value=2),
        discarded=3,
    )

    replaced = original.replace(
        value=4, nested=SemanticStruct(value=5))
    merged = original.merge({'value': 6, 'nested': {'value': 7}})
    shortened = original.without('discarded')

    assert type(replaced) is SemanticStruct
    assert type(replaced.nested) is SemanticStruct
    assert type(merged) is SemanticStruct
    assert type(merged.nested) is SemanticStruct
    assert type(shortened) is SemanticStruct
    assert replaced.value == 4
    assert replaced.nested.value == 5
    assert merged.value == 6
    assert merged.nested.value == 7
    assert 'discarded' not in shortened
    assert original.value == 1
    assert original.nested.value == 2


def test_replace_replaces_nested_fields_whole() -> None:
    original = Struct(branch=Struct(leaf=Struct(value=1)))

    changed_leaf = original.replace(
        branch=Struct(leaf=Struct(value=Struct(next_value=2))))
    changed_branch = original.replace(branch=3)

    assert type(changed_leaf.branch.leaf.value) is Struct
    assert changed_leaf.branch.leaf.value.next_value == 2
    assert changed_branch.branch == 3
    assert original.branch.leaf.value == 1


def test_merge_preserves_structure_at_every_depth() -> None:
    nested_struct = Struct(branch=Struct(leaf=Struct(value=1)))
    nested_value = Struct(branch=Struct(leaf=1))

    with pytest.raises(TypeError, match='incompatible structures'):
        nested_struct.merge({'branch': {'leaf': 2}})
    with pytest.raises(TypeError, match='incompatible structures'):
        nested_value.merge({'branch': {'leaf': Struct(value=2)}})


def test_a_struct_round_trips_through_pickle() -> None:
    """Verifies that `Struct` trees and `Struct` subclasses round-trip through `pickle`."""
    original = Struct(
        gains=SemanticStruct(kp=jnp.asarray(2.0), ki=0.5), count=3)

    loaded = pickle.loads(pickle.dumps(original))

    assert type(loaded) is Struct
    assert type(loaded.gains) is SemanticStruct
    assert loaded.__keys__ == original.__keys__
    assert loaded.gains.kp == 2.0 and loaded.gains.ki == 0.5
    assert loaded.count == 3
