"""Shared task, data, and budget for the tower comparisons.

The task is to predict an exponential moving average of a noisy scalar
sequence. Its decay varies by task. A model adapts on support sequences and is
scored on a query sequence.

Each framework builds the same scale of architecture: an input projection, a
running normalizer, a depth stack of residual RNN cells, and a readout. Three
members form a committee. Time rollout sits inside task-local adaptation,
which sits inside outer meta-training. The inner update remains differentiable
so the outer gradient passes through it.

The files use each framework's native formulation. NodeJAX composes component
transforms whose param and state behavior comes from one contract. Flax NNX
uses graph-aware `vmap`, `scan`, and gradient transforms over Modules, with
`StateAxes` at sites that need an axis policy. The reusable NNX variant factors
those operations into generic functions. Equinox differentiates Module
PyTrees and carries recurrent values explicitly. PyTorch retains the Module
definition while `torch.func` supplies functional fast weights for MAML.

Every main function checks one common convergence criterion: its final loss
must be below 30 percent of its own initial loss. Framework-native
initialization and cell details differ, so the traces are not expected to
match numerically. This family has no timing harness and supports no runtime
or compilation ranking.
"""

import jax
import jax.numpy as jnp

HIDDEN, LAYERS = 8, 2
T = 40
TASKS, K, META_STEPS = 8, 4, 400
INNER_LR, OUTER_LR = 0.05, 3e-3
MOMENTUM = 0.1              # the running norm's stats rate
MEMBERS = 3                     # committee width, mean-mixed


def make_tasks(key: jax.Array):
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
