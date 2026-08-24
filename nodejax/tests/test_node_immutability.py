"""Definition storage stays private while public node views stay small."""

import jax.numpy as jnp
import pytest

from nodejax import Leaf, Node, batch, map_members, parallel
from nodejax.definition import Def
from nodejax.struct import Struct


def _component(methods=(), boundary=()) -> Node:
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return Leaf(
        apply, param=param, name='component', methods=methods,
        boundary=boundary,
    )


def _scale(factor):
    def param():
        return ()

    def apply(param, input):
        return input * factor

    return Leaf(
        apply, param=param, name=f'scale_{factor}')


def test_public_node_does_not_forward_definition_storage():
    component = _component()

    assert type(component._def) is Def
    with pytest.raises(AttributeError):
        component.methods
    with pytest.raises(AttributeError):
        component.boundary
    with pytest.raises(AttributeError):
        component.factory
    with pytest.raises(AttributeError):
        component.statics
    with pytest.raises(AttributeError):
        component.given
    with pytest.raises(AttributeError):
        component.state_given
    with pytest.raises(AttributeError):
        component.member_name
    with pytest.raises(AttributeError):
        component.destructurable
    with pytest.raises(AttributeError):
        component.destructurable_state
    with pytest.raises(AttributeError):
        component.rebuild_members
    with pytest.raises(AttributeError):
        component.preserve_input
    with pytest.raises(AttributeError):
        component.get_apply_input_spec
    with pytest.raises(AttributeError):
        component.replace
    with pytest.raises(AttributeError):
        component._replace


def test_definition_snapshots_external_metadata_without_public_forwarders():
    def read_scale(param):
        return param.scale

    def restart(carried, initialized, decided):
        return initialized

    methods = {'read_scale': read_scale}
    boundaries = {'episode': restart}
    component = _component(methods, boundaries)
    methods['late'] = read_scale
    boundaries['late'] = restart

    assert dict(component._def.methods) == {'read_scale': read_scale}
    assert dict(component._def.boundaries) == {'episode': restart}


def test_tree_surgery_uses_the_private_tree_binding_stage():
    original = batch(_scale(2), n=2)
    rewritten = map_members(
        original,
        lambda node: _scale(10) if node.name == 'scale_2' else node,
    )

    assert jnp.allclose(
        rewritten.parameterize().apply(jnp.ones(2)), 10.0)
    assert jnp.allclose(
        original.parameterize().apply(jnp.ones(2)), 2.0)


def test_nary_tree_surgery_recompiles_from_named_positions():
    branches = parallel(a=_scale(2))
    rewritten = map_members(
        branches,
        lambda node: _scale(10) if node.name == 'scale_2' else node,
    )

    assert rewritten.members.a.name == 'scale_10'
    assert jnp.allclose(
        rewritten.parameterize().apply(a=jnp.ones(1)).a, 10.0)


def test_low_level_definition_without_a_tree_binder_refuses_surgery():
    leaf = _scale(2)
    with pytest.raises(TypeError, match='tree-binding stage'):
        leaf._def.bind_members(Struct())
