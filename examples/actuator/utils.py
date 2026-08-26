"""Small shared math."""

import jax
import jax.numpy as jnp


def wrap(x: jax.Array) -> jax.Array:
    """Map an angle difference to the [-pi, pi] range."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def lerp(a: jax.Array, b: jax.Array, t: float):
    return a + (b - a) * t
