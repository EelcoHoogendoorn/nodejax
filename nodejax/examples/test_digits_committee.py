"""Sequential handwritten-digit classification (sklearn digits, 1797
samples), end to end.

Each 8x8 image is read as a sequence of 8 pixel rows. The model: input
WHITENING (running mean and covariance, ZCA decorrelation, frozen state
at eval), a committee of MEMBERS deep residual RNN encoders (a stack of
LAYERS residual cells with per-member per-layer params and hidden
state), whitening INSIDE the stack as slow state, recurrent state
internalized per forward pass by a mid-pipe scan over the rows, dropout
on the final hidden features (streaming rng-as-state: a new mask every
train step by auto-advance, no key threading), a shared linear head,
and a member-mean vote.

    whiten >> rows >> scan(ensemble(up >> stack(residual(rnn)))) >>
        last >> drop >> head >> mix

MODE: normalizer eval = freezing the state; dropout eval = the SAME
architecture built with rate=0 — a static — with the trained params
bound into it. Param structure is identical across modes, so weights
transfer by bind(). The in-stack whitening reads from a frozen slot
refreshed at episode start by the scan's persist mapping, so eval
logits are per-sample independent (asserted).

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

from nodejax import NodeDef, node_def, serial, ensemble, stack, scan, residual, train_step, ambient, nn
from nodejax.struct import Struct

HIDDEN, MEMBERS, LAYERS = 24, 3, 2
BATCH, EPOCHS = 125, 30


# --- blocks (all shapes input-resolved; no batch size anywhere) ---

def whiten_def(momentum=0.1, eps=1e-2):
    """ZCA input whitening: running mean and COVARIANCE as cyclic state;
    the batch is decorrelated through the inverse matrix square root of
    the running covariance. The eigh runs on STATE, which train_step
    never differentiates — no gradients through degenerate spectra.
    State shapes derive from the input value; eval = freezing the state."""
    def init(ndef):
        features = jax.tree.map(lambda x: x[0], ndef.input)      # one element of the batch
        return Struct(mean=jnp.zeros_like(features),
                      cov=jnp.eye(jnp.shape(features)[0], dtype=features.dtype))

    def apply(state, input):
        centered = input - jnp.mean(input, axis=0)
        batch_cov = centered.T @ centered / input.shape[0]
        new = Struct(mean=(1 - momentum) * state.mean + momentum * jnp.mean(input, axis=0),
                     cov=(1 - momentum) * state.cov + momentum * batch_cov)
        eigvals, eigvecs = jnp.linalg.eigh(state.cov)
        zca = (eigvecs * (1.0 / jnp.sqrt(eigvals + eps))) @ eigvecs.T
        return new, (input - state.mean) @ zca

    return node_def(apply, init=init, name='whiten')


def _inv_sqrt(cov, iters=5):
    """Newton-Schulz inverse matrix square root — smooth, no eigh."""
    eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    scale = jnp.trace(cov)
    y, z = cov / scale, eye
    for _ in range(iters):
        t = 0.5 * (3.0 * eye - z @ y)
        y, z = y @ t, t @ z
    return z / jnp.sqrt(scale)


def layer_whiten_def(momentum=0.05, eps=1e-2):
    """Whitening INSIDE the stack as SLOW STATE, with a two-slot design:
    every read whitens with the FROZEN slot (fixed for the whole
    episode), every step accumulates into the STATS slot. Per-timestep
    read-then-step is not enough under a time scan — step 2 would read
    stats already touched by step 1's batch — so freezing must be
    per-EPISODE: scan's merge refreshes frozen := carried stats at
    episode start. Output therefore depends on persisted state alone:
    eval is per-sample independent, which the test asserts exactly."""
    def init(ndef):
        features = jax.tree.map(lambda x: x[0], ndef.input)
        stats = Struct(mean=jnp.zeros_like(features),
                       cov=jnp.eye(jnp.shape(features)[0], dtype=features.dtype))
        return Struct(frozen=stats, stats=stats)

    def apply(state, input):
        eye = jnp.eye(input.shape[-1], dtype=input.dtype)
        out = (input - state.frozen.mean) @ _inv_sqrt(state.frozen.cov + eps * eye)
        centered = input - jnp.mean(input, axis=0)
        batch_cov = centered.T @ centered / input.shape[0]
        stats = Struct(mean=(1 - momentum) * state.stats.mean + momentum * jnp.mean(input, axis=0),
                       cov=(1 - momentum) * state.stats.cov + momentum * batch_cov)
        return Struct(frozen=state.frozen, stats=stats), out

    return node_def(apply, init=init, name='lwhiten')


def rows_def():
    """(B, 64) image batch -> (8, B, 8) row sequence (time leading)."""
    def apply(input):
        batch_size = input.shape[0]
        return jnp.swapaxes(jnp.reshape(input, (batch_size, 8, 8)), 0, 1)
    return node_def(apply, name='rows')


def build(drop_rate=0.25):
    """The full committee classifier from stock nn blocks plus the local
    whitening nodes. The width and the mode knob are AMBIENT: declared
    at the factories (@ambient), supplied once here — eager construction
    inside one scope, no threading, no generic ceremony. In-shapes
    (row width, head fan-in) derive from the resolved input."""
    with ambient(hidden=HIDDEN, rate=drop_rate):
        layer = layer_whiten_def() >> residual(nn.rnn())   # whiten IN the stack
        cell = nn.linear(HIDDEN) >> stack(layer, n=LAYERS)
        # persist maps write slot <- read slot at episode start: whitening
        # stats carry themselves, the frozen read-copy refreshes from the
        # carried stats, everything unmatched (rnn hidden) re-inits fresh
        encoder = scan(ensemble(cell, n=MEMBERS),
                       persist={'frozen': 'stats', 'stats': 'stats'})
        return serial(
            whiten=whiten_def(),
            rows=rows_def(),
            encoder=encoder,
            last=node_def(lambda input: input[-1], name='last'),
            drop=nn.dropout(),
            head=nn.linear(10),
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
    whitening stats per feature, one dropout stream, nothing else."""
    pipe = build()
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))

    wh = model.param.encoder.stack_lwhiten_res_rnn.res_rnn.wh
    assert wh.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)
    assert not jnp.allclose(wh[0], wh[1])          # independent members
    assert not jnp.allclose(wh[0, 0], wh[0, 1])    # independent layers

    X_train, _, _, _ = data()
    state = pipe.bind(model.param).with_input(X_train[:BATCH]).init(rng=jax.random.PRNGKey(1))
    assert state.whiten.mean.shape == (64,)        # input-resolved, no statics
    assert state.whiten.cov.shape == (64, 64)      # full covariance, not per-feature
    # the encoder's SLOW state (per-member per-layer whitening stats)
    # lives outside the time loop and persists across train steps
    lw = state.encoder.stack_lwhiten_res_rnn.lwhiten
    assert lw.stats.cov.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)
    assert state.drop.rng.shape == (2,)


def test_trains_on_real_digits():
    """End to end: train the committee on 1000 real digits, evaluate the
    rate=0 architecture with the SAME params and the frozen state."""
    X_train, y_train, X_test, y_test = data()
    pipe = build(drop_rate=0.25)
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

    # MODE SWITCH: same params, rate=0 architecture, frozen state
    evaluator = build(drop_rate=0.0).bind(final.model)
    _, logits1 = evaluator.apply(final.inner, X_test)
    _, logits2 = evaluator.apply(final.inner, X_test)
    assert jnp.allclose(logits1, logits2)                    # eval is deterministic

    # ...and TRULY frozen: read-then-step whitening means a sample's
    # logits do not depend on what else is in the eval batch
    _, logits_solo = evaluator.apply(final.inner, X_test[:50])
    assert jnp.allclose(logits_solo, logits1[:50], atol=1e-5)

    test_accuracy = accuracy(logits1, y_test)
    assert test_accuracy > 0.85, test_accuracy

    train_accuracy = accuracy(evaluator.apply(final.inner, X_train)[1], y_train)
    n_weights = sum(leaf.size for leaf in jax.tree.leaves(final.model))
    print(f"\n[digits committee] {n_weights} weights | "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f} over {len(losses)} steps | "
          f"train acc {train_accuracy:.3f} | "
          f"TEST acc {test_accuracy:.3f} ({len(y_test)} unseen digits)")
