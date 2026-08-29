"""Plain loss callables shared by reinforcement-learning examples."""

import jax
import jax.numpy as jnp

from nodejax import PyTree, split_aux


def mse(output: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((output - target) ** 2)


def ensemble_mse(output: PyTree, target: jax.Array) -> jax.Array:
    """Fit every value retained by ``ensemble(...) >> reduce(mean)``."""
    population = split_aux(output)[1].reduce_mean.population
    return jnp.mean((population - target[..., None]) ** 2)
