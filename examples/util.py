"""Grab-bag helpers for the examples and tests: things the DEMONSTRATIONS
share, deliberately not the library's. A loss function is a modeling
choice, so the library ships none; the examples share one spelling."""

from __future__ import annotations

import jax.numpy as jnp


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)
