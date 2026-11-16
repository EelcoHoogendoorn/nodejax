"""Self-supervised Bootstrap Your Own Latent (BYOL) representation learning on `sklearn` digits.

    online:  aug(v1) -> enc -> pred ----.
                                        |-> normalized MSE
    target:  aug(v2) -> enc_ema  -------'

`BYOL` is a three-member `Composite` combining a nested `train_step` trainer over the online
encoder and predictor, an `nn.EMA(tau, warm=True)` filter smoothing the online encoder weights in
state, and a stateless `Views` leaf drawing two augmented views per batch from call-time `rng`.
Custom `param` and `init` functions size the online network from `BYOL`'s input and warm-start the
EMA filter at the initial online encoder weights.

During `apply`, the target branch evaluates `enc.apply(self.ema.state, pair.v2)` outside the
nested trainer's `value_and_grad`, so target representations enter `nmse` as constant target data
without an explicit stop-gradient. Evaluation fits a closed-form ridge classifier on 100 labeled
representations from the EMA encoder and compares accuracy against random initialization while
checking per-dimension embedding spread against collapse.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from nodejax import (
    Composite, Leaf, Node, PNode, nn, node, serial, train_step,
)
from nodejax.struct import Struct

IMAGE, HIDDEN, EMBED = 8, 64, 32
BATCH, STEPS = 125, 1000
TAU = 0.99


ln = Leaf(lambda input: (input - jnp.mean(input, axis=-1, keepdims=True))
              / jnp.sqrt(jnp.var(input, axis=-1, keepdims=True) + 1e-5), name='ln')
# Batch standardization centers each feature across the batch so a constant vector cannot satisfy it.
bstd = Leaf(lambda input: (input - jnp.mean(input, axis=0, keepdims=True))
                / (jnp.std(input, axis=0, keepdims=True) + 1e-5), name='bstd')


def Encoder() -> Node:
    """Builds the per-sample encoder MLP with layer normalization so probe features stay batch-independent."""
    return serial(up=nn.Linear(HIDDEN), ln=ln, act=nn.gelu,
                  emb=nn.Linear(EMBED))


def Predictor() -> Node:
    """Builds the online predictor MLP with batch standardization to oppose representation collapse."""
    return serial(up=nn.Linear(HIDDEN), bn=bstd, act=nn.gelu,
                  down=nn.Linear(EMBED))


def augment(key: jax.Array, batch: jax.Array) -> jax.Array:
    """Applies random wraparound spatial shifts and additive Gaussian noise to flattened `8x8` digits."""
    k_shift, k_noise = jax.random.split(key)
    imgs = batch.reshape(-1, IMAGE, IMAGE)
    shifts = jax.random.randint(k_shift, (imgs.shape[0], 2), -1, 2)

    def roll_one(img: jax.Array, s: jax.Array) -> jax.Array:
        rows = (jnp.arange(IMAGE) - s[0]) % IMAGE
        cols = (jnp.arange(IMAGE) - s[1]) % IMAGE
        return img[rows][:, cols]

    imgs = jax.vmap(roll_one)(imgs, shifts)
    return imgs.reshape(batch.shape) + 0.4 * jax.random.normal(k_noise, batch.shape)


def nmse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean squared error between L2-normalized prediction and target embeddings."""
    p = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean(jnp.sum((p - t) ** 2, axis=-1))


@node
def Views(augment: Callable) -> PNode:
    """Draws two independently augmented views per batch from call-time `rng`."""
    def apply(input, rng) -> Struct:
        return Struct(v1=augment(rng.next(), input),
                      v2=augment(rng.next(), input))

    return Leaf(apply)


@node
def BYOL(enc: Node, pred: Node, augment: Callable,
         opt: optax.GradientTransformation,
         loss: Callable = nmse, tau: float = TAU) -> Node:
    """Executes one BYOL training step over `enc` and `pred` with an EMA target encoder.

    Carries `Struct(train, ema)` in state, evaluating the target branch at `self.ema.state`
    outside `self.train` before updating the EMA filter from the stepped online encoder weights.
    """
    online = serial(enc=enc, pred=pred)
    members = Composite(train=train_step(online, loss, opt),
                        ema=nn.EMA(tau, warm=True),
                        views=Views(augment))

    def param(node, rng) -> Struct:
        """Resolves online network parameter shapes from `BYOL`'s input specification."""
        resolved = train_step(online.with_input(node.input), loss, opt)
        return Struct(
            train=resolved.node.parameterize(rng=rng.next()).param)

    def init(self) -> None:
        """Initializes trainer state and warm-starts `ema` from online encoder weights."""
        self.ema.reset(input=self.train.params().enc)

    def apply(self, input) -> None:
        pair = self.views(input)
        targets = enc.apply(self.ema.state, pair.v2)
        step_loss = self.train(input=pair.v1, target=targets)
        self.ema(self.train.params().enc)
        return step_loss

    return members(apply, param=param, init=init)


def data() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Loads and shuffles `sklearn` digits into 1000 training and 797 held-out examples."""
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def probe_accuracy(feats: jax.Array, labels: jax.Array,
                   feats_test: jax.Array, labels_test: jax.Array) -> jax.Array:
    """Fits a closed-form ridge classifier onto one-hot labels and returns held-out accuracy."""
    A = jnp.concatenate([feats, jnp.ones((len(feats), 1))], axis=1)
    Y = jax.nn.one_hot(labels, 10)
    w = jnp.linalg.solve(A.T @ A + 1e-3 * jnp.eye(A.shape[1]), A.T @ Y)
    A_test = jnp.concatenate([feats_test, jnp.ones((len(feats_test), 1))], axis=1)
    return jnp.mean(jnp.argmax(A_test @ w, axis=-1) == labels_test)


def test_byol_learns_a_representation() -> None:
    """Verifies that BYOL embeddings maintain spread and beat random features under a 100-label ridge probe."""
    X_train, y_train, X_test, y_test = data()
    enc = Encoder()
    labels = 100

    byol = BYOL(enc, Predictor(), augment, optax.adam(1e-3))
    byol = byol.with_input(jnp.zeros((1, IMAGE * IMAGE))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    random_enc_params = byol.state.train.opt.params.model.enc

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(STEPS * BATCH // len(X_train))]
    )[:STEPS * BATCH].reshape(STEPS, BATCH)
    # Composite auxiliary outputs nest the trainer's loss trajectory under its member name `train`.
    byol, (_, aux) = byol.scan(X_train[batch_indices], rng=jax.random.PRNGKey(1))
    losses = aux.train.loss

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]

    feats_train = enc.bind(byol.state.ema).apply(X_train)
    feats_test = enc.bind(byol.state.ema).apply(X_test)

    # Normalized embeddings retain per-dimension variance rather than collapsing to a point.
    spread = jnp.std(feats_test / (jnp.linalg.norm(feats_test, axis=-1, keepdims=True)
                                   + 1e-8), axis=0)
    assert jnp.mean(spread) > 0.05, jnp.mean(spread)

    acc_learned = probe_accuracy(feats_train[:labels], y_train[:labels],
                                 feats_test, y_test)
    acc_random = probe_accuracy(enc.bind(random_enc_params).apply(X_train)[:labels],
                                y_train[:labels],
                                enc.bind(random_enc_params).apply(X_test), y_test)
    acc_pixels = probe_accuracy(X_train[:labels], y_train[:labels], X_test, y_test)
    # Fixed root keys keep the comparison predetermined rather than selected post hoc.
    assert acc_learned > acc_random + 0.005, (acc_learned, acc_random)

    print(f"\n[byol] loss {losses[0]:.3f} -> {losses[-1]:.3f} over {STEPS} steps | "
          f"{labels}-label probe: learned {acc_learned:.3f}, "
          f"random-init {acc_random:.3f}, raw pixels {acc_pixels:.3f} | "
          f"embedding spread {jnp.mean(spread):.3f}")
