"""Verifies `residual` adding a skip connection `x + f(x)` around a Node."""

import jax.numpy as jnp

from nodejax import residual
from nodejax.control import Gain, Integrator


def test_residual():
    r = residual(Gain()).parameterize(scale=jnp.array(2.0))
    assert jnp.allclose(r.apply(3.0), 9.0)

    bound = Gain().parameterize(scale=jnp.array(0.5))
    assert jnp.allclose(residual(bound).apply(2.0), 3.0)


def test_residual_cyclic():
    """Verifies that `residual` threads cyclic inner state while adding the input skip."""
    r = residual(Integrator()).parameterize()
    state = r.init()
    state, out = r.apply(state, 1.0)
    state, out = r.apply(state, 1.0)
    assert jnp.allclose(state, 2.0)
    assert jnp.allclose(out, 3.0)
