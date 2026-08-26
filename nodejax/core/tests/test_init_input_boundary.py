"""Initialization has two definite forms, never a nullable input argument."""

import inspect

import jax.numpy as jnp
import pytest

from nodejax import Leaf, Struct
from nodejax.core.rng import MaybeKeyStream


def _shape_initialized():
    def init(node):
        return jnp.zeros_like(node.input)

    def apply(state, input):
        return state, input

    return Leaf(apply, init=init, name='shape_initialized')


def _value_initialized():
    def init(input):
        return input

    def apply(state, input):
        return state, input

    return Leaf(apply, init=init, name='value_initialized')


def test_compiled_init_has_binary_arity():
    shape_init = _shape_initialized()._def.calls.init
    value_init = _value_initialized()._def.calls.init

    # The canonical order is binding order: what the definition IS, then
    # what it has been bound to, then what the call supplies, entropy last.
    assert not shape_init.requires_input
    assert tuple(inspect.signature(shape_init.impl).parameters) == (
        'definition', 'param', 'formed_input', 'rng')
    assert value_init.requires_input
    assert tuple(inspect.signature(value_init.impl).parameters) == (
        'definition', 'param', 'formed_input', 'input', 'rng')


def test_supplied_input_binds_shape_but_does_not_prime_nonrequiring_init():
    supplied = jnp.asarray([3.0, 4.0, 5.0])

    state = _shape_initialized().init(input=supplied)

    assert state.shape == supplied.shape
    assert jnp.allclose(state, jnp.zeros_like(supplied))


def test_required_input_is_rejected_at_contract_boundary_when_omitted():
    node = _value_initialized().node

    with pytest.raises(TypeError, match='real input value'):
        node.contract.init((), Struct(), MaybeKeyStream())


def test_required_input_reaches_priming_init_as_real_data():
    supplied = jnp.asarray([3.0, 4.0, 5.0])

    state = _value_initialized().init(input=supplied)

    assert jnp.allclose(state, supplied)
