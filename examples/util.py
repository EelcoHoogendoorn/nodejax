"""Provides shared loss helpers used across worked examples and tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes the mean squared error between `pred` and `target`."""
    return jnp.mean((pred - target) ** 2)
