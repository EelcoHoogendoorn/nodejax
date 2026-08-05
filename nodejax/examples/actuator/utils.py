"""Small shared math."""

import jax.numpy as jnp


def wrap(x):
    """Map an angle difference to the [-pi, pi] range."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def lerp(a, b, t):
    return a + (b - a) * t
