"""Sequential handwritten-digit classification (sklearn digits, 1797
samples), end to end.

Each 8x8 image is read as a sequence of 8 pixel rows. The model: input
WHITENING (running mean and covariance, ZCA decorrelation), a committee
of MEMBERS deep residual RNN encoders (a stack of LAYERS normed residual
cells with per-member per-layer params and hidden state), recurrent
state internalized per forward pass by a mid-pipe scan over the rows,
dropout on the final hidden features (streaming rng-as-state: a new mask
every train step by auto-advance, no key threading), a shared linear
head, and a member-mean vote.

    whiten >> rows >> scan(ensemble(up >> stack(norm >> residual(rnn)))) >>
        last >> drop >> head >> mix

Every block is written PER-SAMPLE and batch() adds the axis once, at the
top: no hand-threaded batch dimension anywhere, and the moments the
whitening needs are collectives over the named batch axis.

MODE: normalizer eval = reusing the state; dropout eval = the SAME
architecture built with rate=0 — a static — with the trained params
bound into it. Param structure is identical across modes, so weights
transfer by bind(). Whitening is read-then-step, so a sample's logits do
not depend on what else is in the eval batch (asserted).

State REFRESH RATES are their own subject, in
test_update_granularity.py.

The training loop is also a node: train_step(model) scanned over the
shuffled epoch stream. One key at parameterize splits into every member
and layer; one key at init seeds the whitening offsets and the dropout
stream.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from nodejax import (NodeDef, node_def, serial, ensemble, stack, scan, residual,
                     batch, train_step, tree_freeze, tree_filter, ambient, nn)
from nodejax.struct import Struct

HIDDEN, MEMBERS, LAYERS = 24, 3, 2
BATCH, EPOCHS = 125, 30


# --- blocks (all shapes input-resolved; no batch size anywhere) ---

def rows():
    """One 64-pixel image -> (8, 8): eight rows, time leading."""
    return node_def(lambda input: jnp.reshape(input, (8, 8)), name='rows')


def build(drop_rate=0.25):
    """The committee classifier from stock nn blocks, written PER-SAMPLE:
    one image in, ten logits out. The caller adds the batch axis with
    batch(), and can reach the members first — which is how eval freezes
    the whitening. The width and the mode knob are AMBIENT: declared at
    the factories (@ambient), supplied once here, no threading. In-shapes
    (row width, head fan-in) derive from the resolved input."""
    with ambient(hidden=HIDDEN, rate=drop_rate):
        layer = nn.LayerNorm() >> residual(nn.RNN())
        cell = nn.Linear(HIDDEN) >> stack(layer, n=LAYERS)
        # the recurrent carry is FAST state: internalized by this scan and
        # re-initialized per forward pass, so nothing about one image's
        # sequence leaks into the next
        encoder = scan(ensemble(cell, n=MEMBERS))
        return serial(
            whiten=nn.Whiten(),
            rows=rows(),
            encoder=encoder,
            last=node_def(lambda input: input[-1], name='last'),
            drop=nn.Dropout(),
            head=nn.Linear(10),
            mix=node_def(lambda input: jnp.mean(input, axis=0), name='mix'),
        )


# --- real data ---

def data():
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def xent(pred, target):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, target))


def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def test_assembly():
    """One key -> every member/layer draw; state census matches the story:
    whitening stats per feature (batch-invariant, so one copy), a dropout
    stream per sample, nothing else."""
    pipe = batch(build())
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))

    wh = model.param.encoder.stack_norm_res_rnn.res_rnn.wh
    assert wh.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)
    assert not jnp.allclose(wh[0], wh[1])          # independent members
    assert not jnp.allclose(wh[0, 0], wh[0, 1])    # independent layers

    X_train, _, _, _ = data()
    state = pipe.bind(model.param).with_input(X_train[:BATCH]).init(rng=jax.random.PRNGKey(1))
    assert state.whiten.mean.shape == (64,)        # input-resolved, no statics
    assert state.whiten.cov.shape == (64, 64)      # full covariance, not per-feature
    # the recurrent carry is FAST state, internalized by the mid-pipe scan:
    # it never appears out here at all
    assert state.encoder == ()                  # an empty slot, no carry out here
    assert state.drop.rng.shape == (BATCH, 2)   # an independent mask stream each


def test_trains_on_real_digits():
    """End to end: train the committee on 1000 real digits, evaluate the
    rate=0 architecture with the SAME params and the frozen state."""
    X_train, y_train, X_test, y_test = data()
    pipe = batch(build(drop_rate=0.25))
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))

    # the epoch stream: shuffled batches, the training loop as one scan
    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)                                     # (steps, BATCH)
    stream = Struct(input=X_train[batch_indices], target=y_train[batch_indices])

    trainer = train_step(model, xent, optax.adam(3e-3))   # resolve what you wrap
    final, losses = trainer.scan(
        trainer.init(model=model.param, rng=jax.random.PRNGKey(1)), stream)

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.3 * losses[0]

    # dropout is alive in the train architecture: same batch, advancing
    # stream, different logits
    trained = pipe.bind(final.model)
    advanced, logits_a = trained.apply(final.inner, X_train[:BATCH])
    _, logits_b = trained.apply(advanced, X_train[:BATCH])
    assert not jnp.allclose(logits_a, logits_b)

    # MODE SWITCH: the rate=0 architecture with the whitening moments FROZEN
    # at what training left them, the same params bound in. Nothing stochastic
    # and nothing accumulating remains, so the evaluator is NON-CYCLIC: a plain
    # function of the images, with no state to thread, hold or re-init.
    evaluator = batch(tree_freeze(build(drop_rate=0.0),
                                  tree_filter(final.inner, 'whiten'))).bind(final.model)
    assert not evaluator.ndef.cyclic

    logits1 = evaluator.apply(X_test)
    logits2 = evaluator.apply(X_test)
    assert jnp.allclose(logits1, logits2)                    # eval is deterministic

    # ...and TRULY frozen: read-then-step whitening means a sample's
    # logits do not depend on what else is in the eval batch
    logits_solo = evaluator.apply(X_test[:50])
    assert jnp.allclose(logits_solo, logits1[:50], atol=1e-5)

    test_accuracy = accuracy(logits1, y_test)
    assert test_accuracy > 0.85, test_accuracy

    train_accuracy = accuracy(evaluator.apply(X_train), y_train)
    n_weights = sum(leaf.size for leaf in jax.tree.leaves(final.model))
    print(f"\n[digits committee] {n_weights} weights | "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f} over {len(losses)} steps | "
          f"train acc {train_accuracy:.3f} | "
          f"TEST acc {test_accuracy:.3f} ({len(y_test)} unseen digits)")
