"""Sequential handwritten-digit classification with an RNN committee and ZCA whitening.

    whiten >> rows >> scanned(ensemble(up >> stack(norm >> residual(rnn)))) >>
        last >> drop >> head >> mix

`build` reads each `8x8` image as an 8-row sequence through `nn.Whiten`, a reshape to `(8, 8)`, a
mid-pipeline `scanned(ensemble(...))` over `MEMBERS` stacked normed residual RNN encoders,
last-step selection, `nn.Dropout`, a shared linear head, and a committee-mean vote. Every Leaf is
authored per sample inside an `ambient` scope supplying `hidden` and dropout `rate`; `scanned`
internalizes the per-image recurrent carry so only whitening statistics remain in outer state, and
`batch(build())` adds the batch axis at the outer boundary.

`train_step` trains the batched committee over shuffled epochs while splitting a single boundary
PRNG key across steps for dropout. `nn.eval_mode` specializes `'*.train'` to `False` and folds
`'running_stats'` state into fixed constants via `tree_freeze`, producing a deterministic
non-cyclic evaluator whose per-sample logits are independent of sibling batch elements.
"""

from __future__ import annotations

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


def build(drop_rate: float = 0.25) -> Node:
    """Builds the per-sample RNN committee classifier mapping `(64,)` digit pixels to `(10,)` logits."""
    with ambient(hidden=HIDDEN, rate=drop_rate):
        layer = nn.LayerNorm() >> residual(nn.RNN())
        cell = nn.Linear(HIDDEN) >> stack(layer, n=LAYERS)
        # `scanned` internalizes the per-image recurrent carry so state resets on every forward pass.
        encoder = scanned(ensemble(cell, n=MEMBERS))
        return serial(
            whiten=nn.Whiten(),
            rows=nn.Reshape((8, 8)),
            encoder=encoder,
            last=Leaf(lambda input: input[-1], name='last'),
            drop=nn.Dropout(),
            head=nn.Linear(10),
            mix=Leaf(lambda input: jnp.mean(input, axis=0), name='mix'),
        )


def data() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Loads and shuffles `sklearn` digits into 1000 training and 797 held-out examples."""
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def xent(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean integer-label softmax cross-entropy loss."""
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, target))


def accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Computes top-1 classification accuracy from logits and integer labels."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def test_assembly() -> None:
    """Verifies ensemble and layer parameter axes and confirms that only whitening statistics appear in state."""
    pipe = batch(build())
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))

    wh = model.param.encoder.stack_norm_res_rnn.res_rnn.wh
    assert wh.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)
    assert not jnp.allclose(wh[0], wh[1])
    assert not jnp.allclose(wh[0, 0], wh[0, 1])

    X_train, _, _, _ = data()
    state = pipe.with_input(X_train[:BATCH]).bind(model.param).init()
    assert state.whiten.mean.shape == (64,)
    assert state.whiten.cov.shape == (64, 64)
    # `scanned` internalizes the recurrent carry and `nn.Dropout` draws masks at `apply` without state.
    assert 'encoder' not in state
    assert 'drop' not in state


def test_trains_on_real_digits() -> None:
    """Trains the RNN committee on digits and verifies deterministic batch-independent evaluation."""
    pipe = batch(build(drop_rate=0.25))
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, xent, optax.adam(3e-3))

    X_train, y_train, X_test, y_test = data()
    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)

    final, aux = trained(trainer).apply(input=X_train[batch_indices],
                                        target=y_train[batch_indices],
                                        rng=jax.random.PRNGKey(1))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0]

    # Distinct PRNG keys draw distinct dropout masks when calling the training-mode `PSNode`.
    _, logits_a = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(3))
    assert not jnp.allclose(logits_a, logits_b)

    # `nn.eval_mode` disables dropout and freezes the accumulated ZCA whitening moments.
    evaluator = nn.eval_mode(final)
    assert not evaluator.cyclic and evaluator.state == ()

    _, logits1 = evaluator(X_test)
    _, logits2 = evaluator(X_test)
    assert jnp.allclose(logits1, logits2)

    # Frozen whitening moments make each sample's logits independent of sibling batch elements.
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
