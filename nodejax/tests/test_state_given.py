"""State supplied by bound members.

A state-bound member contributes finished state to a composition. An
explicit value for that member replaces the stored state; it does not reopen
the member's initializer. Stored keys rekey from the enclosing boundary.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, PNode, PSNode, Leaf, serial
from nodejax.control import Gain, Integrator
from nodejax.struct import Struct


def PrimedLevel() -> Node:
    def apply(state, input):
        return state + input, state
    def init(input):
        return jnp.asarray(input)
    return Leaf(apply, init=init, name='primed_level')


def Streamy() -> Node:
    def init(rng):
        return Struct(rng=rng)
    def apply(state, input):
        return state, input
    return Leaf(apply, init=init, name='streamy')


def test_a_stored_state_fills_the_open_slot():
    bound = Integrator().parameterize().bind(state=jnp.asarray(5.0))
    pipe = serial(a=bound, g=Gain())
    node = pipe.parameterize(g=Struct(scale=2.0))
    state = node.init()              # no replacement or key: the store fills
    assert jnp.allclose(state.a, 5.0)


def test_finished_state_replaces_the_store_without_reopening_init():
    bound = PrimedLevel().parameterize().bind(state=jnp.asarray(7.0))
    node = serial(lv=bound, g=Gain()).parameterize(g=Struct(scale=1.0))
    assert jnp.allclose(node.init().lv, 7.0)
    assert jnp.allclose(node.init(input=jnp.asarray(99.0)).lv, 7.0)
    assert jnp.allclose(node.init(lv=jnp.asarray(1.0)).lv, 1.0)


def test_all_state_bound_composes_state_bound():
    """>> over state-bound members returns a state-bound composite."""
    a = Integrator().parameterize().bind(state=jnp.asarray(0.0))
    b = Integrator().parameterize().bind(state=jnp.asarray(10.0))
    snp = a >> b
    assert isinstance(snp, PSNode)
    res, out = snp(1.0)
    assert jnp.allclose(res.state.integrator, 1.0)
    assert jnp.allclose(res.state.integrator_2, 11.0)


def test_stored_entropy_never_replays():
    """A stored rng stream re-keys from the boundary: the composite owes
    a key, the fresh stream differs from the stored one, and omitting
    the key is a named refusal rather than a silent replay."""
    k0 = jax.random.PRNGKey(7)
    bound = Streamy().parameterize().bind(state=Struct(rng=k0))
    node = serial(s=bound, g=Gain()).parameterize(g=Struct(scale=1.0))
    state = node.init(rng=jax.random.PRNGKey(1))
    assert not jnp.allclose(state.s.rng, k0)
    # the derived spec already demands the key (the stored stream keeps
    # rng REQUIRED); _rekeyed's never-replays refusal is the deep guard
    with pytest.raises(TypeError, match='rng'):
        node.init()


def test_staterize_mirrors_parameterize():
    bound = Integrator().parameterize().initialize()
    assert isinstance(bound, PSNode)
    assert jnp.allclose(bound.state, Integrator().parameterize().init())
