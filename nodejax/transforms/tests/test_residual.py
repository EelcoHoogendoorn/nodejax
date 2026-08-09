"""residual: the skip connection as a transform — x + f(x)."""

import jax.numpy as jnp

from nodejax import residual
from nodejax.control import Gain, Integrator


def test_residual():
    r = residual(Gain()).parameterize(scale=jnp.array(2.0))
    assert jnp.allclose(r.apply(3.0), 9.0)            # x + 2x

    bound = Gain().parameterize(scale=jnp.array(0.5))
    assert jnp.allclose(residual(bound).apply(2.0), 3.0)


def test_residual_cyclic():
    """State rides through the wrap: the integrator accumulates as
    usual, the output gains the skip."""
    r = residual(Integrator()).parameterize(gain=jnp.array(1.0))
    state = r.init()
    state, out = r.apply(state, 1.0)
    state, out = r.apply(state, 1.0)
    assert jnp.allclose(state, 2.0)                   # plain accumulation
    assert jnp.allclose(out, 3.0)                     # 1 + integral
