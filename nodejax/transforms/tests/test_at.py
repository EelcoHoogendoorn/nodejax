"""at: route a node onto one field of a Struct input."""

import jax.numpy as jnp

from nodejax import at
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_at_routes_field():
    node = at(Gain(), 'x').parameterize(scale=jnp.asarray(2.0))
    out = node.apply(Struct(x=jnp.asarray(3.0), y=jnp.asarray(7.0)))
    assert jnp.allclose(out.x, 6.0)              # the field went through the node
    assert jnp.allclose(out.y, 7.0)              # the rest passed through untouched


def test_at_cyclic_state_threads():
    node = at(Integrator(), 'x').parameterize(gain=jnp.asarray(1.0))
    state = node.init()
    state, out = node.apply(state, Struct(x=jnp.asarray(2.0), y=jnp.asarray(9.0)))
    state, out2 = node.apply(state, Struct(x=jnp.asarray(2.0), y=jnp.asarray(9.0)))
    assert jnp.allclose(out.x, 2.0) and jnp.allclose(out2.x, 4.0)   # state accumulates
    assert jnp.allclose(out2.y, 9.0)
