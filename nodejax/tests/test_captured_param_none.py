"""Explicit parameter values override params captured by composition."""

import jax.numpy as jnp

from nodejax import Composite, Leaf, Node, serial
from nodejax.struct import Struct


def NullableScale() -> Node:
    def param(value):
        return value

    def apply(param, input):
        return input if param is None else input * param

    return Leaf(apply, param=param, name='nullable_scale')


def InputAwareNullableScale() -> Node:
    def param(node, value):
        return jnp.asarray(value) + jnp.zeros(node.input.shape).sum()

    def apply(param, input):
        return input if param is None else input * param

    return Leaf(apply, param=param, name='input_aware_nullable_scale')


def StructScale() -> Node:
    def param(config=Struct(scale=2.0)):
        return config

    def apply(param, input):
        if param is None:
            return input
        if 'scale' in param:
            return input * param.scale
        return input + param.offset

    return Leaf(apply, param=param, name='struct_scale')


def InputAwareStructScale() -> Node:
    def param(node, config=Struct(scale=2.0)):
        node.input_shape
        return config

    def apply(param, input):
        return input if param is None else input * param.scale

    return Leaf(apply, param=param, name='input_aware_struct_scale')


def test_serial_explicit_none_replaces_captured_param():
    captured = NullableScale().parameterize(value=2.0)

    model = serial(stage=captured).parameterize(stage=None)

    assert model.param.stage is None
    assert jnp.allclose(model.apply(3.0), 3.0)


def test_authored_wiring_explicit_none_replaces_captured_param():
    input = jnp.asarray(0.0)
    captured = InputAwareNullableScale().with_input(input).parameterize(
        value=2.0)
    members = Composite(stage=captured)

    def apply(self, input):
        return self.stage(input)

    node = members(apply, name='nullable_wiring').with_input(input)
    model = node.parameterize(stage=None)

    assert model.param.stage is None
    assert jnp.allclose(model.apply(3.0), 3.0)


def test_captured_struct_param_is_one_replaceable_value():
    captured = StructScale().parameterize(config=Struct(scale=3.0))
    node = serial(stage=captured)

    default = node.parameterize()
    removed = node.parameterize(stage=None)
    replaced = node.parameterize(stage=Struct(offset=4.0))

    assert default.param.stage.scale == 3.0
    assert removed.param.stage is None
    assert replaced.param.stage.__keys__ == ('offset',)
    assert jnp.allclose(replaced(2.0), 6.0)


def test_captured_struct_param_remains_complete_when_nested():
    captured = StructScale().parameterize(config=Struct(scale=3.0))
    node = serial(inner=serial(stage=captured))

    removed = node.parameterize(inner=Struct(stage=None))

    assert removed.param.inner.stage is None
    assert jnp.allclose(removed(2.0), 2.0)


def test_authored_shape_walk_replaces_captured_struct_whole():
    input = jnp.zeros(3)
    captured = InputAwareStructScale().with_input(input).parameterize(
        config=Struct(scale=3.0))

    def apply(self, input):
        return self.stage(input)

    node = Composite(stage=captured)(
        apply, name='struct_wiring').with_input(input)
    removed = node.parameterize(stage=None)

    assert removed.param.stage is None
    assert jnp.allclose(removed(jnp.ones(3)), jnp.ones(3))
