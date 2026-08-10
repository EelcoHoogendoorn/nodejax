"""batch: vmap over the input axis — params broadcast, state per element."""

import jax
import jax.numpy as jnp

from nodejax import batch, node_def
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_batch():
    b = batch(Gain()).parameterize(scale=jnp.array(2.0))
    out = b.apply(jnp.array([1.0, 2.0, 4.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 8.0]))


def test_batch_cyclic():
    b = batch(Integrator(), n=3).parameterize(gain=jnp.array(1.0))
    state = b.init()
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0]))


def test_batch_single_batch_state():

    def pop_apply(param, state, input):
        total = jax.lax.psum(input, 'batch')
        new_state = Struct(count=state.count + total)
        return new_state, input * param.scale

    def pop_init(param):                      # a constant seed: no shape, no value
        return Struct(count=jnp.array(0.0))

    pop_node = node_def(pop_apply, init=pop_init, param=lambda: Struct(scale=2.0), tags={'single_batch_state'})

    b = batch(pop_node).with_input(jnp.array([1.0, 2.0, 3.0]))
    m = b.parameterize()
    state = m.init()
    assert state.count.shape == ()
    state, out = m.apply(state, jnp.array([1.0, 2.0, 3.0]))
    assert state.count == 6.0
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0]))
