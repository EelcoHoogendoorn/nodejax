"""Provides rotary angle wrapping and linear interpolation helpers for actuator models."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def wrap(angle: jax.Array) -> jax.Array:
    """Wraps `angle` into `[-pi, pi]`."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def lerp(low: jax.Array, high: jax.Array, weight: float | jax.Array) -> jax.Array:
    """Interpolates linearly between `low` and `high` by `weight`."""
    return low + (high - low) * weight
