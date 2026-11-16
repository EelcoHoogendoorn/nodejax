"""Verifies state capture and explicit state replacement on `PSNode` members.

A state-bound member contributes its captured state to an enclosing
composition. Passing an explicit state value for that member replaces the
captured state without re-running `init`.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, PSNode, Leaf, serial
from nodejax.control import Gain, Integrator
from nodejax.struct import Struct


def PrimedLevel() -> Node:
    def apply(state, input):
        return state + input, state
    def init(input):
        return jnp.asarray(input)
    return Leaf(apply, init=init, name='primed_level')


def RandomLevel() -> Node:
    def init(rng):
        return jax.random.normal(rng.next())
    def apply(state, input):
        return state, state + input
    return Leaf(apply, init=init, name='random_level')


def test_a_stored_state_fills_the_open_slot():
    bound = Integrator().parameterize().bind(state=jnp.asarray(5.0))
    pipe = serial(a=bound, g=Gain())
    node = pipe.parameterize(g=Struct(scale=2.0))
    state = node.init()
    assert jnp.allclose(state.a, 5.0)


def test_finished_state_replaces_the_store_without_reopening_init():
    bound = PrimedLevel().parameterize().bind(state=jnp.asarray(7.0))
    node = serial(lv=bound, g=Gain()).parameterize(g=Struct(scale=1.0))
    assert jnp.allclose(node.init().lv, 7.0)
    assert jnp.allclose(node.init(input=jnp.asarray(99.0)).lv, 7.0)
    assert jnp.allclose(node.init(lv=jnp.asarray(1.0)).lv, 1.0)


def test_all_state_bound_composes_state_bound():
    """Verifies `>>` over `PSNode` members returns a state-bound `PSNode` composite."""
    a = Integrator().parameterize().bind(state=jnp.asarray(0.0))
    b = Integrator().parameterize().bind(state=jnp.asarray(10.0))
    snp = a >> b
    assert isinstance(snp, PSNode)
    res, out = snp(1.0)
    assert jnp.allclose(res.state.integrator, 1.0)
    assert jnp.allclose(res.state.integrator_2, 11.0)


def test_captured_stochastic_init_state_needs_no_key():
    """Verifies capturing a `PSNode` from a stochastic `init` makes enclosing `init` deterministic."""
    bound = RandomLevel().parameterize().initialize(rng=jax.random.PRNGKey(7))
    node = serial(s=bound, g=Gain()).parameterize(g=Struct(scale=1.0))
    assert not node.contract.init_takes_rng
    state = node.init()
    assert jnp.allclose(state.s, bound.state)
    with pytest.raises(TypeError, match='does not accept rng'):
        node.init(rng=jax.random.PRNGKey(1))


def test_staterize_mirrors_parameterize():
    bound = Integrator().parameterize().initialize()
    assert isinstance(bound, PSNode)
    assert jnp.allclose(bound.state, Integrator().parameterize().init())
