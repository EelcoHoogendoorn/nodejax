"""stack: scan over the layer axis — per-layer params, layer k feeds k+1."""

import jax.numpy as jnp

from nodejax import stack
from nodejax.control import Gain


def test_stack():
    s = stack(Gain()).parameterize(scale=jnp.array([2.0, 3.0]))
    assert jnp.allclose(s.apply(1.0), 6.0)
