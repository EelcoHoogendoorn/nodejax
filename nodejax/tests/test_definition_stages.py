"""The construction ladder: statics, tree, input, params, then state."""

import jax.numpy as jnp

from nodejax import Leaf, Node, REQUIRED, map_members, node, serial


@node
def Scale(factor: float = 1.0) -> Node:
    def param():
        return ()

    def apply(param, input):
        return factor * input

    return Leaf(apply, param=param, name=f'scale_{factor:g}')


@node
def OpenScale(factor: float) -> Node:
    return Scale(factor)


@node
def Pair(left: float = 1.0, right: float = 2.0) -> Node:
    return serial(a=Scale(left), b=Scale(right))


@node
def Tower(depth: int = 1) -> Node:
    return serial(**{
        f'layer_{index}': Scale(index + 1.0)
        for index in range(depth)
    })


@node
def FromParts(parts) -> Node:
    return serial(**parts)


def test_chained_static_specialization_accumulates():
    base = serial(a=Scale(1.0), b=Scale(2.0))
    left = base.specialize(**{'a.factor': 3.0})
    both = left.specialize(**{'b.factor': 5.0})

    assert both.statics_by_path()['a.factor'] == 3.0
    assert both.statics_by_path()['b.factor'] == 5.0
    assert both.parameterize().apply(jnp.asarray(1.0)) == 15.0


def test_static_specialization_discards_member_surgery_but_keeps_statics():
    tuned = serial(a=Scale(1.0), b=Scale(2.0)).specialize(
        **{'a.factor': 3.0})
    replacement = Scale(7.0)
    variant = map_members(
        tuned,
        lambda member: replacement if member.name == 'scale_2' else member,
    )

    fresh = variant.specialize(**{'a.factor': 4.0})

    assert fresh.statics_by_path()['a.factor'] == 4.0
    assert fresh.statics_by_path()['b.factor'] == 2.0
    assert fresh.parameterize().apply(jnp.asarray(1.0)) == 8.0


def test_empty_specialization_restores_the_canonical_tree():
    base = serial(a=Scale(1.0), b=Scale(2.0))
    replacement = Scale(7.0)
    variant = map_members(
        base,
        lambda member: replacement if member.name == 'scale_2' else member,
    )

    canonical = variant.specialize()

    assert canonical.parameterize().apply(jnp.asarray(1.0)) == 2.0


def test_static_and_tree_replay_discard_input_evidence():
    resolved = Pair().with_input(jnp.zeros(3))

    assert resolved.specialize(left=4.0).contract.input_spec is None
    assert map_members(
        resolved, lambda member: member).contract.input_spec is None


def test_static_replay_may_change_topology():
    one = Tower(depth=1)
    two = one.specialize(depth=2)

    assert one.members.__keys__ == ('layer_0',)
    assert two.members.__keys__ == ('layer_0', 'layer_1')


def test_topology_change_rejects_an_accumulated_path_that_disappears():
    tuned = Tower(depth=2).specialize(**{'layer_1.factor': 7.0})

    try:
        tuned.specialize(depth=1)
    except TypeError as error:
        assert 'layer_1' in str(error)
    else:
        raise AssertionError('a stale accumulated static path must be loud')


def test_generic_records_defaults_before_it_can_build():
    @node
    def Cell(width: int, leak: float = 0.1) -> Node:
        return Scale(leak)

    pending = Cell()

    assert pending.statics_by_path() == {
        'width': REQUIRED,
        'leak': 0.1,
    }


def test_generic_overlay_descends_through_mappings_without_losing_siblings():
    pending = FromParts(parts={
        'a': OpenScale(),
        'b': Scale(2.0),
    })

    built = pending.specialize(**{'parts.a.factor': 3.0})

    assert built.statics_by_path()['parts.a.factor'] == 3.0
    assert built.statics_by_path()['parts.b.factor'] == 2.0
    assert built.parameterize().apply(jnp.asarray(1.0)) == 6.0
