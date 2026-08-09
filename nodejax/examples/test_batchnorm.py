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

from nodejax import NodeDef, serial, batch, freeze, train_step, nn
from nodejax.struct import Struct
from nodejax.examples.test_conv_vit import data, xent, accuracy

WIDTH, MOMENTUM = 32, 0.1
BATCH, EPOCHS = 125, 80


def build() -> NodeDef:
	"""(64,) pixels -> (10,) logits, per-sample, all stock nn; batchnorm
	mid-pipe makes the pipe cyclic and gives it the 'batch' axis need."""
	return serial(
		up=nn.Linear(WIDTH),
		bn1=nn.BatchNorm(MOMENTUM),
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

	batched = batch(pipe).with_input(jnp.zeros((BATCH, 64)))
	model = batched.parameterize(rng=jax.random.PRNGKey(0))

	shuffle = np.random.RandomState(1)
	batch_indices = np.concatenate(
		[shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
	).reshape(-1, BATCH)
	stream = Struct(input=X_train[batch_indices], target=y_train[batch_indices])

	trainer = train_step(batched, xent, optax.adam(3e-3))
	final, losses = trainer.scan(trainer.init(model=model.param), stream)

	assert jnp.all(jnp.isfinite(losses))
	assert losses[-1] < 0.3 * losses[0]

	# the running moments absorbed the training stream: they sit at the
	# train set's activation statistics, far from their (0, 1) init
	up = X_train @ final.model.up.w + final.model.up.b
	assert jnp.allclose(final.inner.bn1.mean, jnp.mean(up, axis=0), rtol=0.1)

	# eval = the frozen state, as a transform: freeze bakes the trained
	# stats in and the def stops being cyclic — nothing to thread. The
	# stats currently live at the training batch size (state-axis
	# affinity, tasks/todo.md, will free them of it), so eval scores a
	# test batch of that size
	trained = batched.bind(final.model)
	frozen = freeze(trained, final.inner)
	logits = frozen.apply(X_test[:BATCH])
	assert jnp.allclose(logits, frozen.apply(X_test[:BATCH]))   # deterministic
	test_accuracy = accuracy(logits, y_test[:BATCH])
	assert test_accuracy > 0.9, test_accuracy

	# the frozen stats matter: freezing FRESH (0, 1) stats instead
	# mis-normalizes every layer
	cold = freeze(trained, trained.init()).apply(X_test[:BATCH])
	assert accuracy(cold, y_test[:BATCH]) < test_accuracy

	print(f"\n[batchnorm] loss {losses[0]:.3f} -> {losses[-1]:.3f} over "
	      f"{len(losses)} steps | TEST acc {test_accuracy:.3f} | "
	      f"cold-stats acc {accuracy(cold, y_test[:BATCH]):.3f}")
