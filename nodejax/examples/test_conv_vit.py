"""A little conv-transformer on sklearn digits: a proper convolutional
stem (3x3 SAME convs, channel growth, a stride-2 downsample), the
feature grid flattened to tokens, hand-coded multi-head attention,
pre-norm blocks, flattened readout.

    image >> conv(1->16) >> gelu >> conv(16->32, /2) >> gelu
          >> tokens >> pos >> stack(block) >> flat >> head

DOGFOOD TARGET — shape-changing stacks. This is the architecture family
where explicit parameter sizing has to prove itself: widths are AMBIENT
(hidden/heads supplied once), channels thread the stem by hand
(1 -> 16 -> 32), and the sizes marked GEOMETRY below are hand-computed
conv arithmetic: the grid side after the strided conv, the token count,
and the flattened readout width (tokens * hidden, the classic
conv-flatten size). Late-binding param specs would absorb all of them;
this file is the measure of whether they hurt enough to build it, and
test_nn_vit is the answer: the same architecture and assertions with
every layer imported from nodejax.nn, no width or geometry anywhere.
EVERYTHING LOCAL HERE IS LOCAL ON PURPOSE, as the baseline of that pair.

Attention is ONE leaf node: a fused qkv matmul (one reshape, unpack the
size-3 dim), softmax, merge, all einsum math inside a single apply,
because composition is for parametric and stateful structure, not for
arithmetic. The transformer block is
structure: residual(norm >> attn) >> residual(norm >> up >> gelu >> down),
stacked with per-layer params by stack(n=DEPTH).

The whole model is stateless (no dropout, no running stats), so the pipe
is non-cyclic: eval is plain apply, and train_step carries a trivial ()
model state — the contract's stateless corner exercised by a full NN.

THE MODEL IS WRITTEN PER-SAMPLE: the conv sees one (H, W, C) image,
attention one (T, hidden) sequence, and batch(pipe) adds the data axis
at the boundary. Contrast the digits committee, where whitening
consumes cross-sample batch statistics and the batch must therefore
ride as data; here every node is per-sample math, so the axis belongs
to a transform.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_digits

from nodejax import (node, trained, scan, PNode, Node, nn, Leaf, serial, stack, residual,
                     batch, train_step, ambient, KeyStream)
from nodejax.struct import Struct

IMAGE, STEM, HIDDEN, HEADS, DEPTH = 8, 16, 32, 4, 2
GRID = IMAGE // 2              # GEOMETRY: SAME padding, one stride-2 conv
TOKENS = GRID * GRID           # GEOMETRY: sequence length seen by attention
BATCH, EPOCHS = 125, 40

@node
def reshape(shape: tuple[int, ...], name: str = 'reshape') -> PNode:
    return Leaf(lambda input: input.reshape(shape), name=name)


# --- the conv stem: real conv layers, channels threaded by hand ---

@node
def Conv(c_in: int, c_out: int, kernel: int = 3, stride: int = 1) -> Node:
    """3x3 SAME convolution over ONE (H, W, C) feature map. lax demands
    a batch dim, so a unit one is added and stripped; under batch() the
    vmap fuses it into a real batched conv."""
    def param(rng: KeyStream) -> Struct:
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, c_out))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in), bias=jnp.zeros(c_out))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(stride, stride), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return out + param.bias

    return Leaf(apply, param=param)


@node(name='pos')
def PosEmbed(tokens: int, hidden: int) -> Node:
    """Learned position embedding over the token axis."""
    def param(rng: KeyStream) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), (tokens, hidden)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input + param.embed

    return Leaf(apply, param=param)


# --- the transformer block ---

@node(name='norm')
def LayerNorm(hidden: int, eps: float = 1e-5) -> Node:
    """Layernorm over the feature axis."""
    def param() -> Struct:
        return Struct(scale=jnp.ones(hidden), bias=jnp.zeros(hidden))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)
        return (input - mean) / jnp.sqrt(var + eps) * param.scale + param.bias

    return Leaf(apply, param=param)


@node(name='attn')
def Attention(hidden: int, heads: int) -> Node:
    """Multi-head self-attention over one (T, hidden) sequence."""
    assert hidden % heads == 0
    dim = hidden // heads

    def param(rng: KeyStream) -> Struct:
        return Struct(
            wqkv=jax.random.normal(rng.next(), (hidden, 3 * hidden)) / jnp.sqrt(hidden),
            wo=jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        qkv = (input @ param.wqkv).reshape(*input.shape[:-1], 3, heads, dim)
        q, k, v = jnp.unstack(qkv, axis=-3)
        logits = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(dim)
        mix = jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), v)
        return mix.reshape(input.shape) @ param.wo

    return Leaf(apply, param=param)


@node
def Linear(n_in: int, n_out: int) -> Node:
    def param(rng: KeyStream) -> Struct:
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)


def Block() -> Node:
    """Pre-norm transformer block; shape-preserving, so stack-able."""
    return serial(
        attn=residual(serial(norm=LayerNorm(), attn=Attention())),
        mlp=residual(serial(norm=LayerNorm(),
                            up=Linear(HIDDEN, 4 * HIDDEN),
                            act=nn.gelu,
                            down=Linear(4 * HIDDEN, HIDDEN))),
    )


def build() -> Node:
    """The per-sample model: (64,) pixels -> (10,) logits."""
    with ambient(hidden=HIDDEN, heads=HEADS, tokens=TOKENS):
        return serial(
            image=reshape((IMAGE, IMAGE, 1), name='image'),
            conv1=Conv(1, STEM),
            act1=nn.gelu,
            conv2=Conv(STEM, HIDDEN, stride=2),   # downsample: grid side halves
            act2=nn.gelu,
            tokens=reshape((TOKENS, HIDDEN), name='tokens'),
            pos=PosEmbed(),
            blocks=stack(Block(), n=DEPTH),
            flat=reshape((-1,), name='flat'),
            head=Linear(TOKENS * HIDDEN, 10),   # GEOMETRY: the conv-flatten size
        )


# --- real data (same split as the digits committee) ---

def data() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    digits = load_digits()
    X = jnp.asarray(digits.data / 16.0, dtype=jnp.float32)
    y = jnp.asarray(digits.target)
    permutation = np.random.RandomState(0).permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X[:1000], y[:1000], X[1000:], y[1000:]


def xent(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, target))


def accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def test_assembly():
    """One key -> every draw; the param census matches the architecture:
    per-layer stacked block params, conv kernel, positions, flat head.
    The per-sample model maps one image; batch() adds the data axis
    without touching the params."""
    pipe = build()
    assert not pipe.cyclic                                   # fully stateless
    model = pipe.parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.conv1.kernel.shape == (3, 3, 1, STEM)
    assert model.param.conv2.kernel.shape == (3, 3, STEM, HIDDEN)
    assert model.param.pos.embed.shape == (TOKENS, HIDDEN)
    wqkv = model.param.blocks.attn.attn.wqkv
    assert wqkv.shape == (DEPTH, HIDDEN, 3 * HIDDEN)         # per-layer, fused qkv
    assert not jnp.allclose(wqkv[0], wqkv[1])                # independent layers
    assert model.param.blocks.mlp.up.w.shape == (DEPTH, HIDDEN, 4 * HIDDEN)
    assert model.param.head.w.shape == (TOKENS * HIDDEN, 10)

    assert model.apply(jnp.zeros(IMAGE * IMAGE)).shape == (10,)     # one sample
    batched = batch(pipe).bind(model.param)                         # same params
    assert batched.apply(jnp.zeros((5, IMAGE * IMAGE))).shape == (5, 10)


def test_trains_on_real_digits():
    """End to end: the little conv-transformer learns real digits through
    the conv stem, the attention stack, and the flattened readout."""
    X_train, y_train, X_test, y_test = data()

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)


    trainer = train_step(
        batch(build()).with_input(X_train[:BATCH]).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        xent, optax.adam(1e-3))
    done, aux = trained(trainer).apply(input=X_train[batch_indices], target=y_train[batch_indices])

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0]

    _, logits = done(X_test)
    test_accuracy = accuracy(logits, y_test)
    assert test_accuracy > 0.85, test_accuracy

    train_accuracy = accuracy(done(X_train)[1], y_train)
    n_weights = sum(leaf.size for leaf in jax.tree.leaves(done.param))
    print(f"\n[conv-vit] {n_weights} weights | "
          f"loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} over {len(aux.loss)} steps | "
          f"train acc {train_accuracy:.3f} | "
          f"TEST acc {test_accuracy:.3f} ({len(y_test)} unseen digits)")
