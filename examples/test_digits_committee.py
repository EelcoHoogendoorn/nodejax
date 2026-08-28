"""Sequential handwritten-digit classification (sklearn digits, 1797
samples), end to end.

Each 8x8 image is read as a sequence of 8 pixel rows. The model: input
WHITENING (running mean and covariance, ZCA decorrelation), a committee
of MEMBERS deep residual RNN encoders (a stack of LAYERS normed residual
cells with per-member per-layer params and hidden state), recurrent
state internalized per forward pass by a mid-pipe scan over the rows,
dropout on the final hidden features (drawing at apply: the pipe
splits its one boundary key toward the drawing member), a shared linear
head, and a member-mean vote.

    whiten >> rows >> scanned(ensemble(up >> stack(norm >> residual(rnn)))) >>
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
shuffled epoch sequence. One key at parameterize splits into every member
and layer; one key rides the training sequence and splits per step
toward the dropout draws.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from nodejax import (Node, Leaf, serial, ensemble, stack, scan, scanned, trained, residual,
                     batch, train_step, tree_filter, ambient, nn)
from nodejax.struct import Struct

HIDDEN, MEMBERS, LAYERS = 24, 3, 2
BATCH, EPOCHS = 125, 30


# --- blocks (all shapes input-resolved; no batch size anywhere) ---

def build(drop_rate: float=0.25) -> Node:
    """The committee classifier from stock nn blocks, written PER-SAMPLE:
    one image in, ten logits out. The caller adds the batch axis with
    batch(), and can reach the members first — which is how eval freezes
    the whitening. The width and the mode knob are AMBIENT: declared at
    the factories (@node), supplied once here, no threading. In-shapes
    (row width, head fan-in) derive from the resolved input."""
    with ambient(hidden=HIDDEN, rate=drop_rate):
        layer = nn.LayerNorm() >> residual(nn.RNN())
        cell = nn.Linear(HIDDEN) >> stack(layer, n=LAYERS)
        # the recurrent carry is FAST state: internalized by this scan and
        # re-initialized per forward pass, so nothing about one image's
        # sequence leaks into the next
        encoder = scanned(ensemble(cell, n=MEMBERS))
        return serial(
            whiten=nn.Whiten(),
            rows=nn.Reshape((8, 8)),      # eight rows, time leading
            encoder=encoder,
            last=Leaf(lambda input: input[-1], name='last'),
            drop=nn.Dropout(),
            head=nn.Linear(10),
            mix=Leaf(lambda input: jnp.mean(input, axis=0), name='mix'),
        )


# --- real data ---

def data():
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def xent(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, target))


def accuracy(logits: jax.Array, labels: jax.Array):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def test_assembly():
    """One key -> every member/layer draw; state census matches the story:
    whitening stats per feature (batch-invariant, so one copy) and
    nothing else, dropout having no state to hold."""
    pipe = batch(build())
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))

    wh = model.param.encoder.stack_norm_res_rnn.res_rnn.wh
    assert wh.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)
    assert not jnp.allclose(wh[0], wh[1])          # independent members
    assert not jnp.allclose(wh[0, 0], wh[0, 1])    # independent layers

    X_train, _, _, _ = data()
    state = pipe.with_input(X_train[:BATCH]).bind(model.param).init()
    assert state.whiten.mean.shape == (64,)        # input-resolved, no statics
    assert state.whiten.cov.shape == (64, 64)      # full covariance, not per-feature
    # the recurrent carry is FAST state, internalized by the mid-pipe scan:
    # it never appears out here at all
    assert 'encoder' not in state               # no slot: the scan internalized the carry
    assert 'drop' not in state                  # draws at apply: no slot either


def test_trains_on_real_digits():
    """End to end: train the committee on 1000 real digits, evaluate the
    rate=0 architecture with the SAME params and the frozen state."""
    # the tree, bound: the dropout committee under its trainer
    pipe = batch(build(drop_rate=0.25))
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, xent, optax.adam(3e-3))

    # the data: EPOCHS shuffled passes over the digits, as one sequence
    X_train, y_train, X_test, y_test = data()
    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)                                     # (steps, BATCH)

    # the run: the training loop as one finalized run, its destination
    # kept for the mode switch below
    final, aux = trained(trainer).apply(input=X_train[batch_indices],
                                        target=y_train[batch_indices],
                                        rng=jax.random.PRNGKey(1))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0]

    # dropout is alive in the train architecture: the train build owes a
    # key per call, and different keys draw different masks (`final` IS
    # the trained model, state-bound; its whitening stats ride along)
    _, logits_a = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(3))
    assert not jnp.allclose(logits_a, logits_b)

    # MODE SWITCH, on one object: eval_mode rebuilds the dropout leaves
    # as identities (their mask streams pruned from the carried state),
    # pins the whitening moments, and rebinds the same params. Nothing
    # stochastic and nothing accumulating remains, so the evaluator is
    # NON-CYCLIC: a plain function of the images.
    evaluator = nn.eval_mode(final)
    assert not evaluator.cyclic and evaluator.state == ()

    _, logits1 = evaluator(X_test)
    _, logits2 = evaluator(X_test)
    assert jnp.allclose(logits1, logits2)                    # eval is deterministic

    # ...and TRULY frozen: read-then-step whitening means a sample's
    # logits do not depend on what else is in the eval batch
    _, logits_solo = evaluator(X_test[:50])
    assert jnp.allclose(logits_solo, logits1[:50], atol=1e-5)

    test_accuracy = accuracy(logits1, y_test)
    assert test_accuracy > 0.85, test_accuracy

    train_accuracy = accuracy(evaluator(X_train)[1], y_train)
    n_weights = sum(leaf.size for leaf in jax.tree.leaves(final.param))
    print(f"\n[digits committee] {n_weights} weights | "
          f"loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} over {len(aux.loss)} steps | "
          f"train acc {train_accuracy:.3f} | "
          f"TEST acc {test_accuracy:.3f} ({len(y_test)} unseen digits)")
