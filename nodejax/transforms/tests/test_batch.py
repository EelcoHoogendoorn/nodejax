"""batch: vmap over the input axis — params broadcast, state per element."""

import jax.numpy as jnp

from nodejax import batch
from nodejax.examples import gain_def, integrator_def


def test_batch():
    b = batch(gain_def()).parameterize(scale=jnp.array(2.0))
    out = b.apply(jnp.array([1.0, 2.0, 4.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 8.0]))


def test_batch_cyclic():
    b = batch(integrator_def(), n=3).parameterize(gain=jnp.array(1.0))
    state = b.init()
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0]))
