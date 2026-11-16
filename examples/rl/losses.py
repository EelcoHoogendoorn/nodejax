"""Provides scalar loss and discounted return functions for reinforcement learning."""

import jax
import jax.numpy as jnp


def mse(output: jax.Array, target: jax.Array) -> jax.Array:
    """Computes the mean squared error between `output` and `target`."""
    return jnp.mean((output - target) ** 2)


def discounted_backward_sum(
    increment: jax.Array,
    factor: float,
    final: jax.Array,
) -> jax.Array:
    """Accumulates `increment[..., t] + factor * result[..., t + 1]` backward along the trailing time axis from `final`."""
    def step(carry, current):
        value = current + factor * carry
        return value, value

    values = jax.lax.scan(step, final, jnp.moveaxis(increment, -1, 0), reverse=True)[1]
    return jnp.moveaxis(values, 0, -1)


def td_lambda(
    cost: jax.Array,
    next_value: jax.Array,
    *,
    discount: float,
    trace: float,
) -> jax.Array:
    """Computes TD-lambda targets along the trailing time axis."""
    return discounted_backward_sum(
        cost + discount * (1.0 - trace) * next_value,
        discount * trace,
        next_value[..., -1],
    )


def bootstrapped_costs(
    cost: jax.Array,
    terminals: jax.Array,
    *,
    discount: float,
) -> jax.Array:
    """Sums discounted running `cost` along the trailing time axis plus discounted `terminals`."""
    n_steps = cost.shape[-1]
    discounts = discount ** jnp.arange(n_steps)
    return jnp.sum(discounts * cost, axis=-1) + discount ** n_steps * terminals
