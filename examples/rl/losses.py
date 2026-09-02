"""Plain loss callables shared by reinforcement-learning examples."""

import jax
import jax.numpy as jnp


def mse(output: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((output - target) ** 2)


def td_lambda(
    cost: jax.Array,
    next_value: jax.Array,
    *,
    discount: float,
    trace: float,
) -> jax.Array:
    """TD-lambda targets for a time-major world batch."""
    def step(carry, input):
        current_cost, current_value = input
        target = current_cost + discount * (
            (1.0 - trace) * current_value + trace * carry
        )
        return target, target

    target = jax.lax.scan(
        step,
        next_value[-1],
        (cost, next_value),
        reverse=True,
    )[1]
    return target


def bootstrapped_costs(
    cost: jax.Array,
    terminals: jax.Array,
    *,
    discount: float,
) -> jax.Array:
    """Discounted running costs plus discounted terminal values."""
    n_steps = cost.shape[0]
    discounts = discount ** jnp.arange(n_steps)
    return (
        jnp.sum(discounts[:, None] * cost, axis=0)
        + discount ** n_steps * terminals
    )
