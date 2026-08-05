"""Minimal BYOL on sklearn digits: self-supervised representation
learning with an EMA target network.

    online:  aug(v1) -> enc -> pred ----.
                                        |-> normalized MSE
    target:  aug(v2) -> enc_ema  -------'      (gradient-free side)

THE TARGET NETWORK IS A LOW-PASS FILTER ON WEIGHTS: ema_def is a
one-pole filter over an arbitrary pytree — the same node a simulation
would put on a sensor signal — and feeding it the online encoder's
param subtree IS the target network. The BYOL step is one user-land
cyclic node, generic over encoder/predictor/augmentation/optimizer,
whose state holds its collaborators' states: the inner trainer's
(train_step nests — a trainer is a node, so its state rides inside a
bigger state), the ema filter's (the smoothed encoder params,
warm-started as a copy), and the view maker's (augmentation entropy as
rng-as-state, auto-advanced inside that node — the byol wiring itself
touches no keys).
Stop-gradient comes for free: the target features are computed outside
the trainer's value_and_grad and arrive through the loss's target
argument as data.

WHY RAW WIRING INSTEAD OF composite(): the target path re-binds the
encoder def against a sibling's STATE every step (enc.bind(state.ema) —
the smoothed weights ARE the params). A composite member's params are a
fixed slice of the composite's param tree, so this param-from-state
crossover is wiring only the raw contract expresses. The price is
structural opacity: rewriters (taps/freeze/map_members) dispatch on
Composite and see this node as a leaf, while everything contract-shaped
(scan, pipes, nesting as a member) works unchanged. Cheap here: byol's
weights live in state, out of rewriter reach either way.

Evaluation is a closed-form ridge probe on frozen features, scored
against the same probe on a random-init encoder — the learned
representation must beat random features, and the embedding must not
have collapsed (per-dimension spread stays healthy).
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from typing import Callable

from nodejax import Node, NodeDef, node_def, serial, train_step, nn, KeyStream
from nodejax.struct import Struct

IMAGE, HIDDEN, EMBED = 8, 64, 32
BATCH, STEPS = 125, 600
TAU = 0.99


ln = node_def(lambda input: (input - jnp.mean(input, axis=-1, keepdims=True))
              / jnp.sqrt(jnp.var(input, axis=-1, keepdims=True) + 1e-5), name='ln')
# batch standardization: the anti-collapse pressure — per-batch centering
# makes a constant embedding impossible (layernorm cannot: it is
# per-sample, and a collapsed embedding satisfies it exactly)
bstd = node_def(lambda input: (input - jnp.mean(input, axis=0, keepdims=True))
                / (jnp.std(input, axis=0, keepdims=True) + 1e-5), name='bstd')


def enc_def() -> NodeDef:
    """Per-sample math only, so probe features are batch-independent."""
    return serial(up=nn.linear(HIDDEN), ln=ln, act=nn.gelu,
                  emb=nn.linear(EMBED))


def pred_def() -> NodeDef:
    """The predictor is train-time only, so batch standardization lives
    here — collapse pressure where it works, eval untouched."""
    return serial(up=nn.linear(HIDDEN), bn=bstd, act=nn.gelu,
                  down=nn.linear(EMBED))


def augment(key: jax.Array, batch: jax.Array) -> jax.Array:
    """Random per-sample image shifts (wraparound) plus pixel noise."""
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
    """BYOL's loss: MSE between l2-normalized embeddings (2 - 2 cos)."""
    p = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean(jnp.sum((p - t) ** 2, axis=-1))


def views_def(augment: Callable) -> Node:
    """Two independently augmented views of each batch. Owns the
    augmentation entropy as rng-as-state, auto-advanced per step."""
    def init(rng: jax.Array) -> Struct:
        return Struct(rng=rng)

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, Struct]:
        # multi-draw step: wrap the pre-split key in a scope-local stream
        rng = KeyStream(state.rng)
        return state, Struct(v1=augment(rng.next(), input),
                             v2=augment(rng.next(), input))

    return node_def(apply, init=init, name='views')


def byol_def(enc: NodeDef, pred: NodeDef, augment: Callable,
             opt: optax.GradientTransformation,
             loss: Callable = nmse, tau: float = TAU) -> Node:
    """The BYOL step, generic over encoder, predictor, augmentation and
    optimizer — a wiring of four collaborators: the view maker (owns the
    entropy), the target-side encoder read at the ema filter's state,
    the nested trainer, and the ema filter over the online encoder
    params. state = Struct(train=..., ema=..., views=...)."""
    online = serial(enc=enc, pred=pred)
    trainer = train_step(online, loss, opt)
    smooth = nn.ema(tau)
    views = views_def(augment)

    def init(rng: KeyStream, ndef) -> Struct:
        # a declared rng arrives as a KeyStream; keys cross the member
        # boundaries via next() — the stream stays scope-local
        model = online.with_input(ndef.input).parameterize(rng=rng.next())
        return Struct(train=trainer.init(model=model.param),
                      ema=smooth.init(input=model.param.enc),
                      views=views.init(rng=rng.next()))

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        new_views, pair = views.apply(state.views, input)
        targets = enc.apply(state.ema, pair.v2)
        new_train, step_loss = trainer.apply(
            state.train, Struct(input=pair.v1, target=targets))
        new_ema, _ = smooth.apply(state.ema, new_train.model.enc)
        return Struct(train=new_train, ema=new_ema, views=new_views), step_loss

    return node_def(apply, init=init, name='byol')


# --- data and probe ---

def data() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def probe_accuracy(feats: jax.Array, labels: jax.Array,
                   feats_test: jax.Array, labels_test: jax.Array) -> jax.Array:
    """Closed-form ridge regression onto one-hot labels."""
    A = jnp.concatenate([feats, jnp.ones((len(feats), 1))], axis=1)
    Y = jax.nn.one_hot(labels, 10)
    w = jnp.linalg.solve(A.T @ A + 1e-3 * jnp.eye(A.shape[1]), A.T @ Y)
    A_test = jnp.concatenate([feats_test, jnp.ones((len(feats_test), 1))], axis=1)
    return jnp.mean(jnp.argmax(A_test @ w, axis=-1) == labels_test)


def test_byol_learns_a_representation():
    """Probed in the few-label regime (100 labels), where representations
    pay: the learned features beat both a random-init encoder and raw
    pixels under the same probe, and the embedding does not collapse.
    (With all 1000 labels a raw-pixel probe wins on digits — self-
    supervision buys label efficiency, not magic.)"""
    X_train, y_train, X_test, y_test = data()
    enc = enc_def()
    labels = 100

    byol = byol_def(enc, pred_def(), augment, optax.adam(1e-3))
    state = byol.with_input(jnp.zeros((1, IMAGE * IMAGE))).init(rng=jax.random.PRNGKey(0))
    random_enc_params = state.train.model.enc      # the untrained encoder

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(STEPS * BATCH // len(X_train))]
    )[:STEPS * BATCH].reshape(STEPS, BATCH)
    final, losses = byol.scan(state, X_train[batch_indices])

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]

    feats_train = enc.bind(final.ema).apply(X_train)
    feats_test = enc.bind(final.ema).apply(X_test)

    # no collapse: the embedding keeps per-dimension spread
    spread = jnp.std(feats_test / (jnp.linalg.norm(feats_test, axis=-1, keepdims=True)
                                   + 1e-8), axis=0)
    assert jnp.mean(spread) > 0.05, jnp.mean(spread)

    acc_learned = probe_accuracy(feats_train[:labels], y_train[:labels],
                                 feats_test, y_test)
    acc_random = probe_accuracy(enc.bind(random_enc_params).apply(X_train)[:labels],
                                y_train[:labels],
                                enc.bind(random_enc_params).apply(X_test), y_test)
    acc_pixels = probe_accuracy(X_train[:labels], y_train[:labels], X_test, y_test)
    assert acc_learned > acc_random + 0.02, (acc_learned, acc_random)
    assert acc_learned > acc_pixels, (acc_learned, acc_pixels)

    print(f"\n[byol] loss {losses[0]:.3f} -> {losses[-1]:.3f} over {STEPS} steps | "
          f"{labels}-label probe: learned {acc_learned:.3f}, "
          f"random-init {acc_random:.3f}, raw pixels {acc_pixels:.3f} | "
          f"embedding spread {jnp.mean(spread):.3f}")
