"""Focused lowering tests for authored leaves."""

import jax.numpy as jnp
import pytest

from nodejax import Leaf
from nodejax.contract import ApplyCall, InitCall, ParamCall
from nodejax.definition import Def
from nodejax.struct import Struct


def test_leaf_decorator_with_arguments_builds_the_component():
    @Leaf(name='twice')
    def twice(input):
        return 2.0 * input

    assert twice.name == 'twice'
    assert twice(jnp.asarray(3.0)) == 6.0


def test_leaf_lowers_natural_signatures_to_canonical_calls():
    seen = {}

    def param(node, scale):
        seen['param_node'] = node
        return Struct(scale=jnp.asarray(scale))

    def init(param):
        return jnp.zeros_like(param.scale)

    def apply(param, state, input):
        state = state + param.scale * input
        return state, state

    component = Leaf(
        apply, param=param, init=init, name='accumulator')
    definition = component._def

    assert type(definition) is Def
    assert type(definition.calls.param) is ParamCall
    assert type(definition.calls.init) is InitCall
    assert type(definition.calls.apply) is ApplyCall
    assert definition.calls.param.form.declaration.__keys__ == ('scale',)
    assert definition.calls.init.form.declaration.__keys__ == ()
    assert not definition.calls.init.requires_input
    assert definition.calls.apply.form.declaration.__keys__ == ('input',)


def test_authored_node_channel_is_a_restricted_view():
    observed = {}

    def param(node):
        observed['node'] = node
        observed['input'] = node.input
        observed['input_spec'] = node.input_spec
        observed['input_shape'] = node.input_shape
        return ()

    def apply(param, input):
        return input

    component = Leaf(apply, param=param).with_input(jnp.zeros(3))
    component.parameterize()

    assert observed['input'].shape == (3,)
    assert observed['input_spec'].shape == (3,)
    assert observed['input_shape'] == (3,)
    with pytest.raises(AttributeError):
        observed['node'].contract
    with pytest.raises(AttributeError):
        observed['node'].members


def test_struct_valued_constructor_defaults_are_complete_values():
    param_default = Struct(scale=2.0)
    state_default = Struct(count=3.0)

    def param(config=param_default):
        return config

    def init(param, initial=state_default):
        return initial

    def apply(param, state, input):
        return state, Struct(param=param, state=state, input=input)

    component = Leaf(apply, param=param, init=init)
    default = component.parameterize().initialize()
    assert default.param.__keys__ == ('scale',)
    assert default.param.scale == 2.0
    assert default.state.__keys__ == ('count',)
    assert default.state.count == 3.0

    replaced = component.parameterize(
        config=Struct(offset=4.0)).initialize(
            initial=Struct(total=5.0))
    assert replaced.param.__keys__ == ('offset',)
    assert replaced.param.offset == 4.0
    assert replaced.state.__keys__ == ('total',)
    assert replaced.state.total == 5.0


def test_authored_struct_runtime_input_is_one_required_value():
    component = Leaf(lambda input: input)
    input = Struct(left=jnp.asarray(1.0), right=jnp.asarray(2.0))

    output = component(input)

    assert output.__keys__ == ('left', 'right')
    assert output.left == 1.0
    assert output.right == 2.0


def test_constructors_define_owned_roles_even_when_apply_does_not_read_them():
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(start):
        return jnp.asarray(start)

    source = Leaf(
        lambda input: input * 2.0,
        param=param,
        init=init,
        name='owned_roles',
    )

    assert source.parametric
    assert source.cyclic
    bound = source.parameterize(scale=3.0)
    state = bound.init(start=4.0)
    next_state, output = bound.apply(state, 5.0)
    assert next_state == 4.0
    assert output == 10.0


def test_leaf_apply_cannot_name_an_absent_owned_role():
    with pytest.raises(TypeError, match='no param constructor exists'):
        Leaf(lambda param, input: input, name='missing_param')

    with pytest.raises(TypeError, match='no initializer exists'):
        Leaf(lambda state, input: (state, input), name='missing_state')


def test_leaf_self_is_not_supported():
    with pytest.raises(TypeError, match='leaf apply does not accept self'):
        Leaf(lambda self, input: input, name='self_apply')

    def init(self):
        return self

    with pytest.raises(TypeError, match='leaf init does not accept self'):
        Leaf(lambda input: input, init=init, name='self_init')
