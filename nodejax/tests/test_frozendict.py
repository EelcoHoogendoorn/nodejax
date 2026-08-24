import pickle

import pytest

from nodejax.frozendict import frozendict


def test_frozendict_has_immutable_mapping_semantics():
    values = frozendict({'first': 1}, second=2)

    assert tuple(values) == ('first', 'second')
    assert dict(values) == {'first': 1, 'second': 2}
    assert values == {'first': 1, 'second': 2}
    with pytest.raises(TypeError):
        values['third'] = 3


def test_frozendict_supports_copy_union_hash_and_pickle():
    values = frozendict({'first': 1, 'second': 2})

    assert values.copy() == values
    assert values | {'third': 3} == {
        'first': 1, 'second': 2, 'third': 3,
    }
    assert {'zero': 0} | values == {
        'zero': 0, 'first': 1, 'second': 2,
    }
    assert frozendict.fromkeys(('a', 'b'), 4) == {'a': 4, 'b': 4}
    assert hash(values) == hash(frozendict({'second': 2, 'first': 1}))
    assert pickle.loads(pickle.dumps(values)) == values


def test_frozendict_is_unhashable_when_a_value_is_unhashable():
    with pytest.raises(TypeError):
        hash(frozendict(items=[]))
