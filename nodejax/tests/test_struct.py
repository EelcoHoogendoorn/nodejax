"""Value behavior of NodeJAX's immutable record."""

import pytest

from nodejax.struct import Struct


class SemanticStruct(Struct):
    """A Struct subtype whose type carries meaning to its consumer."""


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
