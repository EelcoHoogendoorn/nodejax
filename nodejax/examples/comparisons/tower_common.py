"""Shared task, data, and budget for the tower comparisons.

Everything the three framework files have in common lives here, so
what remains in each file is exactly what differs: the model spelling
and the state routing. The task: predict an exponential moving average
of a noisy scalar sequence, chosen because the target is recurrent by
construction. The meta task family varies the decay per task; a model
adapts on K support sequences and is scored on a query sequence.
"""

import jax
import jax.numpy as jnp

HIDDEN, LAYERS = 8, 2
B, T, STEPS = 16, 40, 800
TASKS, K, META_STEPS, INNER_LR = 8, 4, 400, 0.05


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def make_data(key):
    xs = jax.random.normal(key, (B, T))

    def ema(carry, x):
        y = 0.9 * carry + 0.1 * x
        return y, y

    _, ys = jax.lax.scan(ema, jnp.zeros(B), xs.T)
    return xs, ys.T


def make_tasks(key):
    """Support and query sequences for TASKS tasks, each an EMA with its
    own decay."""
    k1, k2, k3 = jax.random.split(key, 3)
    alphas = jax.random.uniform(k1, (TASKS,), minval=0.6, maxval=0.95)
    sup_x = jax.random.normal(k2, (TASKS, K, T))
    qry_x = jax.random.normal(k3, (TASKS, T))

    def ema(alpha, xs):
        def cell(carry, x):
            y = alpha * carry + (1 - alpha) * x
            return y, y
        return jax.lax.scan(cell, 0.0, xs)[1]

    sup_y = jax.vmap(lambda a, xs: jax.vmap(lambda s: ema(a, s))(xs))(alphas, sup_x)
    qry_y = jax.vmap(ema)(alphas, qry_x)
    return sup_x, sup_y, qry_x, qry_y
