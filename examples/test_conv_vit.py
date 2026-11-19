"""Per-sample convolutional vision transformer on `8x8` `sklearn` digits with explicit geometry.

    image >> conv(1->16) >> gelu >> conv(16->32, /2) >> gelu
          >> tokens >> pos >> stack(block) >> flat >> head

A two-layer `3x3` convolutional stem grows image channels (`1 -> 16 -> 32`) and halves spatial
resolution with a stride-2 layer before reshaping the `4x4` grid into `16` tokens, adding learned
positional embeddings, passing through a `stack` of pre-norm multi-head self-attention blocks, and
classifying from the flattened token table. Shared dimensions (`hidden`, `heads`, `tokens`) flow
through an `ambient` scope while spatial flatten sizes derive from the stride geometry.

Every Leaf owns parameters without state, making the composed pipeline stateless (`not pipe.cyclic`).
`batch(pipe)` maps the per-sample network across a data batch without altering parameter shapes,
and `train_step` optimizes it with an empty `()` model state.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from nodejax import (Generic, node, trained, PNode, Node, nn, Leaf, serial, stack, residual,
                     batch, train_step, ambient)
from nodejax.struct import Struct

IMAGE, STEM, HIDDEN, HEADS, DEPTH = 8, 16, 32, 4, 2
# One stride-2 `SAME` convolution halves the spatial grid side.
GRID = IMAGE // 2
# Flattening the `GRID x GRID` feature map produces `TOKENS` attention positions.
TOKENS = GRID * GRID
BATCH, STEPS = 32, 3


@node
def reshape(shape: tuple[int, ...], name: str = 'reshape') -> PNode:
    """Reshapes an input array to `shape` as a parameter-free Leaf."""
    return Leaf(lambda input: input.reshape(shape), name=name)


@node
def Conv(c_in: int, c_out: int, kernel: int = 3, stride: int = 1) -> Node:
    """Applies a 2D `SAME`-padded convolution and bias over a single `(H, W, C)` feature map."""
    def param(rng) -> Struct:
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, c_out))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in), bias=jnp.zeros(c_out))

    def apply(param, input) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(stride, stride), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return out + param.bias

    return Leaf(apply, param=param)


@node(name='pos')
def PosEmbed(tokens: int, hidden: int) -> Node:
    """Adds a learned `(tokens, hidden)` positional embedding table to a token sequence."""
    def param(rng) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), (tokens, hidden)))

    def apply(param, input) -> jax.Array:
        return input + param.embed

    return Leaf(apply, param=param)


@node(name='norm')
def LayerNorm(hidden: int, eps: float = 1e-5) -> Node:
    """Normalizes activations along the trailing feature axis of width `hidden`."""
    def param() -> Struct:
        return Struct(scale=jnp.ones(hidden), bias=jnp.zeros(hidden))

    def apply(param, input) -> jax.Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)
        return (input - mean) / jnp.sqrt(var + eps) * param.scale + param.bias

    return Leaf(apply, param=param)


@node(name='attn')
def Attention(hidden: int, heads: int) -> Node:
    """Computes multi-head scaled dot-product self-attention over a `(T, hidden)` sequence."""
    assert hidden % heads == 0
    dim = hidden // heads

    def param(rng) -> Struct:
        return Struct(
            wqkv=jax.random.normal(rng.next(), (hidden, 3 * hidden)) / jnp.sqrt(hidden),
            wo=jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden))

    def apply(param, input) -> jax.Array:
        qkv = (input @ param.wqkv).reshape(*input.shape[:-1], 3, heads, dim)
        q, k, v = jnp.unstack(qkv, axis=-3)
        logits = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(dim)
        mix = jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), v)
        return mix.reshape(input.shape) @ param.wo

    return Leaf(apply, param=param)


@node
def Linear(n_in: int, n_out: int) -> Node:
    """Projects trailing features from `n_in` to `n_out` with scaled Gaussian weights and zero bias."""
    def param(rng) -> Struct:
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param, input) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)


def Block() -> Node | Generic:
    """Assembles a shape-preserving pre-norm transformer block with residual attention and MLP sublayers."""
    return serial(
        attn=residual(serial(norm=LayerNorm(), attn=Attention())),
        mlp=residual(serial(norm=LayerNorm(),
                            up=Linear(HIDDEN, 4 * HIDDEN),
                            act=nn.gelu,
                            down=Linear(4 * HIDDEN, HIDDEN))),
    )


def build() -> Node:
    """Builds the per-sample convolutional vision transformer mapping `(64,)` pixels to `(10,)` logits."""
    with ambient(hidden=HIDDEN, heads=HEADS, tokens=TOKENS):
        return serial(
            image=reshape((IMAGE, IMAGE, 1), name='image'),
            conv1=Conv(1, STEM),
            act1=nn.gelu,
            conv2=Conv(STEM, HIDDEN, stride=2),
            act2=nn.gelu,
            tokens=reshape((TOKENS, HIDDEN), name='tokens'),
            pos=PosEmbed(),
            blocks=stack(Block(), n=DEPTH),
            flat=reshape((-1,), name='flat'),
            head=Linear(TOKENS * HIDDEN, 10),
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
    """Verifies parameter shapes across the stem, stacked transformer blocks, and batched wrapper."""
    pipe = build()
    assert not pipe.cyclic
    model = pipe.parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.conv1.kernel.shape == (3, 3, 1, STEM)
    assert model.param.conv2.kernel.shape == (3, 3, STEM, HIDDEN)
    assert model.param.pos.embed.shape == (TOKENS, HIDDEN)
    wqkv = model.param.blocks.attn.attn.wqkv
    assert wqkv.shape == (DEPTH, HIDDEN, 3 * HIDDEN)
    assert not jnp.allclose(wqkv[0], wqkv[1])
    assert model.param.blocks.mlp.up.w.shape == (DEPTH, HIDDEN, 4 * HIDDEN)
    assert model.param.head.w.shape == (TOKENS * HIDDEN, 10)

    assert model.apply(jnp.zeros(IMAGE * IMAGE)).shape == (10,)
    batched = batch(pipe).bind(model.param)
    assert batched.apply(jnp.zeros((5, IMAGE * IMAGE))).shape == (5, 10)


def test_trains_on_real_digits() -> None:
    """Trains the convolutional vision transformer on digits and verifies parameter updates and evaluation shapes."""
    X_train, y_train, X_test, y_test = data()

    shuffle = np.random.RandomState(1)
    batch_indices = shuffle.randint(0, len(X_train), size=(STEPS, BATCH))

    initial = batch(build()).with_input(X_train[:BATCH]).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(initial, xent, optax.adam(1e-3))
    done, aux = trained(trainer).apply(input=X_train[batch_indices], target=y_train[batch_indices])

    assert aux.loss.shape == (STEPS,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, done.param, initial.param))

    logits = done(X_test[:BATCH])
    assert logits.shape == (BATCH, 10)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.isfinite(accuracy(logits, y_test[:BATCH]))
