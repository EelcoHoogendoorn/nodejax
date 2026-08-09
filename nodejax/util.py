"""Small helpers shared across the test suite and examples."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def tile(tree, n):
    """Tile a pytree along a new leading axis (for scanning a fixed batch)."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)
