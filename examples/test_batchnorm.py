"""Batch normalization on handwritten digits using running moments as cyclic `Node` state.

Each `nn.BatchNorm` leaf normalizes activations by its stored `mean` and `var` before updating
those moments with exponential factor `MOMENTUM`, and `train_step` threads the running statistics
alongside optimizer state across batches.

Because `nn.BatchNorm` pools moments across the named `'batch'` collective axis supplied by
`batch(...)` and declares `'single_batch_state'`, the model is authored per sample while keeping a
single shared state copy across the batch. Freezing the trained `PSNode` with `tree_freeze` folds
those accumulated moments into fixed constants so evaluation becomes non-cyclic and deterministic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import trained, Node, serial, batch, tree_freeze, train_step, nn
from examples.test_conv_vit import data, xent

WIDTH, MOMENTUM = 32, 0.1
BATCH, STEPS = 32, 4


def build() -> Node:
    """Builds a per-sample two-hidden-layer MLP with `nn.BatchNorm` on `(64,)` digit vectors."""
    return serial(
        up=nn.Linear(WIDTH),
        batchnorm1=nn.BatchNorm(MOMENTUM),
        act1=nn.gelu,
        mid=nn.Linear(WIDTH),
        bn2=nn.BatchNorm(MOMENTUM),
        act2=nn.gelu,
        head=nn.Linear(10),
    )


def test_batchnorm_trains_and_freezes() -> None:
    """Trains a batch-normalized MLP on digits and verifies running-moment updates and frozen determinism."""
    X_train, y_train, X_test, _ = data()
    pipe = build()
    assert pipe.cyclic

    shuffle = np.random.RandomState(1)
    batch_indices = shuffle.randint(0, len(X_train), size=(STEPS, BATCH))

    initial = batch(pipe).with_input(jnp.zeros((BATCH, 64))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(initial, xent, optax.adam(3e-3))
    final, aux = trained(trainer).apply(input=X_train[batch_indices],
                                        target=y_train[batch_indices])

    assert aux.loss.shape == (STEPS,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, initial.param))

    # Running moments move away from their zero initialization across training batches.
    assert not jnp.array_equal(
        final.state.batchnorm1.mean,
        jnp.zeros_like(final.state.batchnorm1.mean),
    )

    # Freezing the state-bound model absorbs its accumulated moments and makes evaluation deterministic.
    frozen = tree_freeze(final)
    assert not frozen.cyclic
    logits = frozen(X_test[:BATCH])
    assert jnp.allclose(logits, frozen(X_test[:BATCH]))

    # Freezing fresh (0, 1) moments on the same trained weights alters evaluation logits.
    cold = tree_freeze(final.pnode.initialize())(X_test[:BATCH])
    assert not jnp.allclose(cold, logits)
