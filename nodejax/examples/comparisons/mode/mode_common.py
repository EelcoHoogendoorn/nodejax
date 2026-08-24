"""Shared task, data and checks for the mode comparison: TRAIN AND EVAL
as a property of the program, not a flag on it.

The model carries the two classic mode-dependent members: dropout, alive
in training and gone at eval, and a batch norm, stats accumulating in
training and frozen at eval. Every column trains the same model on the
same blobs and then produces its evaluator, and the comparison is HOW:
what code changes between the modes. torch flips a bit on the object,
flax flips two flags, haiku threads an is_training argument through
every call, equinox rewrites flag leaves and hands the norm state
around, and nodejax builds the eval architecture: the rate-0 build with
the same params bound and the stats frozen, no flag existing anywhere.

Three checks, printed per column:

    train_stochastic    two train-mode passes on one batch differ
    eval_deterministic  two eval passes on one batch are identical
    eval_isolated       a sample's eval logits do not depend on what
                        else is in the batch: the stats truly froze

Scores are sanity checks; the comparison is structural. The rival files
carry their own copy of the generator on purpose: each stays runnable
as one self-contained file in an environment without this package.
"""

import numpy as np
import jax
import jax.numpy as jnp

DIM, HIDDEN, CLASSES = 8, 16, 3
N, STEPS = 128, 200
RATE, MOMENTUM, LR = 0.3, 0.1, 0.02


def make_data(rs: np.random.RandomState):
    """CLASSES gaussian blobs in DIM dimensions, one batch."""
    centers = 2.0 * rs.standard_normal((CLASSES, DIM))
    labels = rs.randint(CLASSES, size=N)
    xs = centers[labels] + rs.standard_normal((N, DIM))
    return jnp.asarray(xs, jnp.float32), jnp.asarray(labels, jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


def report(name: str, train_stochastic: bool, eval_deterministic: bool,
           eval_isolated: bool, first: float, last: float) -> None:
    print(f'{name:10s} train_stochastic={train_stochastic} '
          f'eval_deterministic={eval_deterministic} eval_isolated={eval_isolated} '
          f'loss {first:.2f} -> {last:.2f}', flush=True)
