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

from nodejax import trained, scan, Node, serial, batch, tree_freeze, train_step, nn
from nodejax.struct import Struct
from examples.test_conv_vit import data, xent, accuracy

WIDTH, MOMENTUM = 32, 0.1
BATCH, EPOCHS = 125, 80


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
	"""Trains a batch-normalized MLP on digits and verifies frozen-statistic evaluation."""
	X_train, y_train, X_test, y_test = data()
	pipe = build()
	assert pipe.cyclic

	shuffle = np.random.RandomState(1)
	batch_indices = np.concatenate(
		[shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
	).reshape(-1, BATCH)


	trainer = train_step(
		batch(pipe).with_input(jnp.zeros((BATCH, 64))).parameterize(
			rng=jax.random.PRNGKey(0)).initialize(),
		xent, optax.adam(3e-3))
	final, aux = trained(trainer).apply(input=X_train[batch_indices],
	                                    target=y_train[batch_indices])

	assert jnp.all(jnp.isfinite(aux.loss))
	assert aux.loss[-1] < 0.3 * aux.loss[0]

	# Running moments converge to the training-set pre-activation statistics.
	up = X_train @ final.param.up.w + final.param.up.b
	assert jnp.allclose(final.state.batchnorm1.mean, jnp.mean(up, axis=0), rtol=0.1)

	# Freezing the state-bound model absorbs its learned moments and makes evaluation deterministic.
	frozen = tree_freeze(final)
	_, logits = frozen(X_test[:BATCH])
	assert jnp.allclose(logits, frozen(X_test[:BATCH])[1])
	test_accuracy = accuracy(logits, y_test[:BATCH])
	assert test_accuracy > 0.9, test_accuracy

	# Freezing fresh (0, 1) moments on the same trained weights degrades accuracy.
	_, cold = tree_freeze(final.pnode.initialize())(X_test[:BATCH])
	assert accuracy(cold, y_test[:BATCH]) < test_accuracy

	print(f"\n[batchnorm] loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} over "
	      f"{len(aux.loss)} steps | TEST acc {test_accuracy:.3f} | "
	      f"cold-stats acc {accuracy(cold, y_test[:BATCH]):.3f}")
