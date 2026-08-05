"""Batchnorm on digits: running statistics as ordinary cyclic state.

Batchnorm is the classic hard state problem for functional NN
libraries: statistics that update across training steps, are computed
over the batch, and must freeze for evaluation — the usual source of
mode flags, mutable collections and train/eval forks. Here the running
moments are simply the node's state: training threads them because
train_step threads model state, and evaluation is applying the trained
state and discarding the returned update — freezing is a call-site
decision, not a mode.

Cross-sample statistics dictate the data layout. THE MODEL IS WRITTEN
BATCHED: a per-sample model under batch() would give every sample its
own private running moments, so the batch axis rides as data (as in
the digits committee, and in contrast to the per-sample conv-vit).
Where the batch axis lives is the one real design decision batchnorm
forces; the state handling asks nothing.

The node normalizes by its running moments — not the batch moments —
and updates them afterwards: train and eval then run the exact same
function, with eval merely not keeping the returned state.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import NodeDef, serial, train_step, nn
from nodejax.struct import Struct
from nodejax.examples.test_conv_vit import data, xent, accuracy

WIDTH, MOMENTUM = 32, 0.1
BATCH, EPOCHS = 125, 80


def build() -> NodeDef:
	"""(batch, 64) pixels -> (batch, 10) logits, all stock nn;
	batchnorm mid-pipe makes the whole pipe cyclic."""
	return serial(
		up=nn.linear(WIDTH),
		bn1=nn.batch_norm(MOMENTUM),
		act1=nn.gelu,
		mid=nn.linear(WIDTH),
		bn2=nn.batch_norm(MOMENTUM),
		act2=nn.gelu,
		head=nn.linear(10),
	)


def test_batchnorm_trains_and_freezes():
	"""Train through the running stats, then evaluate against them
	frozen: the stats are learned equipment, carried as state."""
	X_train, y_train, X_test, y_test = data()
	pipe = build()
	assert pipe.cyclic
	model = pipe.with_input(jnp.zeros((1, 64))).parameterize(rng=jax.random.PRNGKey(0))

	shuffle = np.random.RandomState(1)
	batch_indices = np.concatenate(
		[shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
	).reshape(-1, BATCH)
	stream = Struct(input=X_train[batch_indices], target=y_train[batch_indices])

	trainer = train_step(pipe, xent, optax.adam(3e-3))
	final, losses = trainer.scan(trainer.init(model=model.param), stream)

	assert jnp.all(jnp.isfinite(losses))
	assert losses[-1] < 0.3 * losses[0]

	# the running moments absorbed the training stream: they sit at the
	# train set's activation statistics, far from their (0, 1) init
	trained = pipe.bind(final.model)
	up = X_train @ final.model.up.w + final.model.up.b
	assert jnp.allclose(final.inner.bn1.mean, jnp.mean(up, axis=0), rtol=0.1)

	# eval = the same apply against the frozen state; the returned
	# update is discarded, so evaluation is deterministic and
	# batch-size independent
	_, logits = trained.apply(final.inner, X_test)
	_, again = trained.apply(final.inner, X_test)
	assert jnp.allclose(logits, again)
	test_accuracy = accuracy(logits, y_test)
	assert test_accuracy > 0.9, test_accuracy

	# the frozen stats matter: the same weights against a fresh (0, 1)
	# state mis-normalize every layer
	_, cold = trained.apply(trained.init(), X_test)
	assert accuracy(cold, y_test) < test_accuracy

	print(f"\n[batchnorm] loss {losses[0]:.3f} -> {losses[-1]:.3f} over "
	      f"{len(losses)} steps | TEST acc {test_accuracy:.3f} | "
	      f"cold-stats acc {accuracy(cold, y_test):.3f}")
