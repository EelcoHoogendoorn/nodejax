"""Batchnorm on digits: running statistics as ordinary cyclic state.

Batchnorm is the classic hard state problem for functional NN
libraries: statistics that update across training steps, are computed
over the batch, and must freeze for evaluation — the usual source of
mode flags, mutable collections and train/eval forks. Here the running
moments are simply the node's state: training threads them because
train_step threads model state, and evaluation is applying the trained
state and discarding the returned update — freezing is a call-site
decision, not a mode.

Cross-sample statistics used to dictate the data layout: without a
named axis, a per-sample model under batch() would give every sample
its own PRIVATE running moments, so the model had to be written
batched. The named axis dissolves that. The model is per-sample like
every nn block, batch() binds the reserved 'batch' axis, the norm's
moments are collectives across it, and the per-element state copies
agree by construction — replicated, never divergent — so any one row
IS the statistics.

The node normalizes by its running moments — not the batch moments —
and updates them afterwards: train and eval then run the exact same
function, with eval merely not keeping the returned state.
"""

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
	"""(64,) pixels -> (10,) logits, per-sample, all stock nn; batchnorm
	mid-pipe makes the pipe cyclic and gives it the 'batch' axis need."""
	return serial(
		up=nn.Linear(WIDTH),
		batchnorm1=nn.BatchNorm(MOMENTUM),
		act1=nn.gelu,
		mid=nn.Linear(WIDTH),
		bn2=nn.BatchNorm(MOMENTUM),
		act2=nn.gelu,
		head=nn.Linear(10),
	)


def test_batchnorm_trains_and_freezes():
	"""Train through the running stats, then evaluate against them
	frozen: the stats are learned equipment, carried as state."""
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

	# the running moments absorbed the training sequence: they sit at the
	# train set's activation statistics, far from their (0, 1) init
	up = X_train @ final.param.up.w + final.param.up.b
	assert jnp.allclose(final.state.batchnorm1.mean, jnp.mean(up, axis=0), rtol=0.1)

	# eval = the binding frozen whole: `final` IS the trained model,
	# state-bound, and tree_freeze consumes the state it holds — the node
	# stops being cyclic, nothing to thread. The stats currently live at
	# the training batch size (state-axis affinity, tasks/todo.md, will
	# free them of it), so eval scores a test batch of that size
	frozen = tree_freeze(final)
	_, logits = frozen(X_test[:BATCH])
	assert jnp.allclose(logits, frozen(X_test[:BATCH])[1])      # deterministic
	test_accuracy = accuracy(logits, y_test[:BATCH])
	assert test_accuracy > 0.9, test_accuracy

	# the frozen stats matter: freezing FRESH (0, 1) stats instead
	# mis-normalizes every layer (the ladder end to end: the param view,
	# a fresh initialize, the freeze)
	_, cold = tree_freeze(final.pnode.initialize())(X_test[:BATCH])
	assert accuracy(cold, y_test[:BATCH]) < test_accuracy

	print(f"\n[batchnorm] loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} over "
	      f"{len(aux.loss)} steps | TEST acc {test_accuracy:.3f} | "
	      f"cold-stats acc {accuracy(cold, y_test[:BATCH]):.3f}")
