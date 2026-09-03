"""Minimal BYOL on sklearn digits: self-supervised representation
learning with an EMA target network.

    online:  aug(v1) -> enc -> pred ----.
                                        |-> normalized MSE
    target:  aug(v2) -> enc_ema  -------'      (gradient-free side)

THE TARGET NETWORK IS A LOW-PASS FILTER ON WEIGHTS: EMA is a
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

from nodejax import (
    Composite, KeyStream, Leaf, Node, PNode, nn, node, serial, train_step,
)
from nodejax.struct import Struct

IMAGE, HIDDEN, EMBED = 8, 64, 32
BATCH, STEPS = 125, 1000
TAU = 0.99


ln = Leaf(lambda input: (input - jnp.mean(input, axis=-1, keepdims=True))
              / jnp.sqrt(jnp.var(input, axis=-1, keepdims=True) + 1e-5), name='ln')
# batch standardization: the anti-collapse pressure — per-batch centering
# makes a constant embedding impossible (layernorm cannot: it is
# per-sample, and a collapsed embedding satisfies it exactly)
bstd = Leaf(lambda input: (input - jnp.mean(input, axis=0, keepdims=True))
                / (jnp.std(input, axis=0, keepdims=True) + 1e-5), name='bstd')


def Encoder() -> Node:
    """Per-sample math only, so probe features are batch-independent."""
    return serial(up=nn.Linear(HIDDEN), ln=ln, act=nn.gelu,
                  emb=nn.Linear(EMBED))


def Predictor() -> Node:
    """The predictor is train-time only, so batch standardization lives
    here — collapse pressure where it works, eval untouched."""
    return serial(up=nn.Linear(HIDDEN), bn=bstd, act=nn.gelu,
                  down=nn.Linear(EMBED))


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


@node
def Views(augment: Callable) -> PNode:
    """Two independently augmented views of each batch. Owns the
    augmentation entropy as rng-as-state, auto-advanced per step."""
    def init(rng: jax.Array) -> Struct:
        return Struct(rng=rng)

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, Struct]:
        # multi-draw step: wrap the pre-split key in a scope-local stream
        rng = KeyStream(state.rng)
        return state, Struct(v1=augment(rng.next(), input),
                             v2=augment(rng.next(), input))

    return Leaf(apply, init=init)


@node
def BYOL(enc: Node, pred: Node, augment: Callable,
             opt: optax.GradientTransformation,
             loss: Callable = nmse, tau: float = TAU) -> PNode:
    """The BYOL step, generic over encoder, predictor, augmentation and
    optimizer — a wiring of four collaborators: the view maker (owns the
    entropy), the target-side encoder read at the ema filter's state,
    the nested trainer, and the ema filter over the online encoder
    params. state = Struct(train=..., ema=..., views=...)."""
    online = serial(enc=enc, pred=pred)
    members = Composite(train=train_step(online, loss, opt),
                   ema=nn.EMA(tau, warm=True),   # target starts AT the online weights
                   views=Views(augment))

    def param(node, rng: KeyStream) -> Struct:
        """The online model reads its width from BYOL's own input.

        The member walk cannot get there by itself: the trainer's input is
        built inside apply from a view, and a view is made by another member,
        so resolving the trainer by walking would need the walk already done.
        Hand-wiring it is what a custom ctor is for, and the tree it returns
        is the member union like any other."""
        resolved = train_step(online.with_input(node.input), loss, opt)
        return Struct(
            train=resolved.node.parameterize(rng=rng.next()).param)

    def init(param, rng: KeyStream) -> Struct:
        """One slot per member, which is what a composite's state IS. Written
        out because two of the three want something the walk cannot provide:
        the ema starts AT the online encoder's weights, which live in the
        trainer's param, and the view maker wants a key of its own."""
        train_state = members.train.bind(param.train).init()
        return Struct(
            train=train_state,
            ema=members.ema.init(input=train_state.opt.params.model.enc),
            views=members.views.init(rng=rng.next()))

    def apply(self, input: jax.Array):
        pair = self.views(input)
        targets = enc.apply(self.ema.state, pair.v2)
        step_loss = self.train(input=pair.v1, target=targets)
        self.ema(self.train.state.opt.params.model.enc)
        return step_loss

    return members(apply, param=param, init=init)


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
    pay: the learned features beat a random-init encoder under the same
    probe, and the embedding does not collapse. Raw pixels remain a useful
    reference, not a per-initialization guarantee."""
    X_train, y_train, X_test, y_test = data()
    enc = Encoder()
    labels = 100

    byol = BYOL(enc, Predictor(), augment, optax.adam(1e-3))
    # weights are PARAMS, drawn at parameterize; the view maker's key is
    # STATE, drawn at init. As a leaf both came out of one init
    byol = byol.with_input(jnp.zeros((1, IMAGE * IMAGE))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize(rng=jax.random.PRNGKey(1))
    random_enc_params = byol.state.train.opt.params.model.enc  # the untrained encoder

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(STEPS * BATCH // len(X_train))]
    )[:STEPS * BATCH].reshape(STEPS, BATCH)
    # the trainer is a MEMBER now, so what it sows arrives under its name:
    # a composite re-emits (output, Aux(<aux by member name>))
    byol, (_, aux) = byol.scan(X_train[batch_indices])
    losses = aux.train.loss

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]

    feats_train = enc.bind(byol.state.ema).apply(X_train)
    feats_test = enc.bind(byol.state.ema).apply(X_test)

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
    # Keep the original predetermined initialization; the test must not pick
    # a favorable root key after comparing trajectories.
    assert acc_learned > acc_random + 0.005, (acc_learned, acc_random)

    print(f"\n[byol] loss {losses[0]:.3f} -> {losses[-1]:.3f} over {STEPS} steps | "
          f"{labels}-label probe: learned {acc_learned:.3f}, "
          f"random-init {acc_random:.3f}, raw pixels {acc_pixels:.3f} | "
          f"embedding spread {jnp.mean(spread):.3f}")
