"""nodejax.nn — stock neural blocks with in-shapes inferred.

The statics taxonomy these blocks follow:

- DESIGN DECISIONS, explicit arguments: out-widths, head counts,
  ratios, kernels — the choices that define an architecture, written
  where the architecture is written (equinox/pytorch style).
- CONSEQUENCES of upstream, derived at parameterize: fan-ins, token
  counts, flatten widths. Constructors read shape through the ndef
  channel (`def param(ndef, rng)`), and a pipe bound to an input spec
  threads each member its own upstream shape — the walk. Shapes live
  in the param values; defs stay shape-generic and reusable.
- Nothing global. Design arguments shared across a whole build
  (hidden, vocab) can be filled by ambient scope — the factories are
  @ambient-decorated, explicit arguments always win, and outside any
  scope they are ordinary functions of ordinary arguments.

    model = (nn.conv(16) >> nn.gelu >> nn.conv(32, stride=2) >> nn.gelu
             >> nn.tokens() >> nn.pos_embed()
             >> stack(nn.block(width=32, heads=4, ratio=4), n=2)
             >> nn.flat >> nn.linear(10))
    node = model.with_input(jnp.zeros((28, 28, 1))).parameterize(rng=key)

Blocks are written per-sample (batch() adds the data axis) and in the
declared tier, the house style.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream
from nodejax.compose import serial
from nodejax.transforms.residual import residual


gelu = node_def(lambda input: jax.nn.gelu(input), name='gelu')

flat = node_def(lambda input: input.reshape(-1), name='flat')


@ambient
def linear(n_out: int):
    """Affine map to n_out features; fan-in from the offer."""
    def param(ndef, rng: KeyStream) -> Struct:
        n_in = ndef.apply_input_spec.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return node_def(apply, param=param, name='linear')


@ambient
def layer_norm(eps: float = 1e-5):
    """Normalize over the feature axis; width from the offer."""
    def param(ndef) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        return Struct(scale=jnp.ones(width), bias=jnp.zeros(width))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)
        return (input - mean) / jnp.sqrt(var + eps) * param.scale + param.bias

    return node_def(apply, param=param, name='norm')


@ambient
def mlp(width: int, ratio: int):
    """The transformer's two-layer feed-forward: expand to ratio*width,
    gelu, project back to width. Plain composition of stock linears;
    width is the design decision, fan-ins derive from the offer."""
    return linear(ratio * width) >> gelu >> linear(width)


@ambient
def attention(heads: int):
    """Multi-head self-attention over one (T, width) sequence,
    width-preserving; width from the offer, split over heads."""
    def param(ndef, rng: KeyStream) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        if width % heads:
            raise ValueError(f'attention: width {width} not divisible by {heads} heads')
        return Struct(
            wqkv=jax.random.normal(rng.next(), (width, 3 * width)) / jnp.sqrt(width),
            wo=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        dim = input.shape[-1] // heads
        qkv = (input @ param.wqkv).reshape(*input.shape[:-1], 3, heads, dim)
        q, k, v = jnp.unstack(qkv, axis=-3)
        logits = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(dim)
        mix = jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), v)
        return mix.reshape(input.shape) @ param.wo

    return node_def(apply, param=param, name='attn')


@ambient
def block(width: int, heads: int, ratio: int):
    """Pre-norm transformer block at an explicit width; width-preserving,
    so stack-able."""
    return serial(
        attn=residual(layer_norm() >> attention(heads)),
        mlp=residual(layer_norm() >> mlp(width, ratio)),
    )


@ambient
def conv(features: int, kernel: int = 3, stride: int = 1):
    """SAME convolution over one (H, W, C) map; in-channels from the
    offer. lax demands a batch dim, so a unit one is added and
    stripped; under batch() the vmap fuses it into a real batched
    conv."""
    def param(ndef, rng: KeyStream) -> Struct:
        c_in = ndef.apply_input_spec.shape[-1]
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, features))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in),
                      bias=jnp.zeros(features))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(stride, stride),
            padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return out + param.bias

    return node_def(apply, param=param, name='conv')


def tokens():
    """(H, W, C) feature map -> (H*W, C) token sequence."""
    return node_def(lambda input: input.reshape(-1, input.shape[-1]), name='tokens')


def pos_embed():
    """Learned position embedding; the whole (T, width) shape from
    the offer."""
    def param(ndef, rng: KeyStream) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), ndef.apply_input_spec.shape))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input + param.embed

    return node_def(apply, param=param, name='pos')


@ambient
def ema(tau: float):
    """One-pole low-pass over an arbitrary pytree: state is the smoothed
    tree, warm-started as a copy of the first offer. On a signal it is
    an ordinary smoothing filter; on a param pytree it is a target
    network."""
    def init(input):
        return input

    def apply(state, input):
        new = jax.tree.map(lambda s, i: tau * s + (1.0 - tau) * i, state, input)
        return new, new

    return node_def(apply, init=init, name='ema')


@ambient
def dropout(rate: float):
    """Dropout as a streaming stochastic node: rate is a STATIC (mode =
    which architecture you built; eval is the rate=0 build with the same
    params bound), the mask stream is rng STATE — a new mask every train
    step by auto-advance, no key threading."""
    def init(rng) -> Struct:
        return Struct(rng=rng)

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        if rate == 0.0:
            return state, input
        keep = jax.random.bernoulli(state.rng, 1.0 - rate, jnp.shape(input))
        return state, jnp.where(keep, input / (1.0 - rate), 0.0)

    return node_def(apply, init=init, name='drop')


@ambient
def batch_norm(momentum: float, eps: float = 1e-5):
    """Batchnorm over a (B, width) activation, batch-written: the
    moments are population quantities, so the batch axis rides as
    data. Normalizes by the RUNNING moments carried as state, then
    folds the batch's moments in — train threads the returned state,
    eval discards it; freezing is a call-site decision, not a mode.
    Width from the offer; gamma/beta restore the affine freedom the
    normalization removes."""
    def param(ndef) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        return Struct(gamma=jnp.ones(width), beta=jnp.zeros(width))

    def init(param: Struct) -> Struct:
        return Struct(mean=jnp.zeros_like(param.beta), var=jnp.ones_like(param.gamma))

    def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        out = (input - state.mean) / jnp.sqrt(state.var + eps) * param.gamma + param.beta
        new = Struct(mean=(1 - momentum) * state.mean + momentum * jnp.mean(input, axis=0),
                     var=(1 - momentum) * state.var + momentum * jnp.var(input, axis=0))
        return new, out

    return node_def(apply, init=init, param=param, name='bn')


@ambient
def embed(vocab: int, hidden: int):
    """Token ids -> vectors by table lookup."""
    def param(rng: KeyStream) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.weight[input]

    return node_def(apply, param=param, name='embed')


@ambient
def unembed(vocab: int, hidden: int):
    """Vectors -> vocab logits through a (vocab, hidden) matrix,
    transposed. Declares the same param structure as embed, so
    tie(pipe, 'embed', 'unembed') shares one matrix across both ends —
    the tied pipe never materializes this slot."""
    def param(rng: KeyStream) -> Struct:
        return Struct(weight=0.3 * jax.random.normal(rng.next(), (vocab, hidden)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.weight.T

    return node_def(apply, param=param, name='unembed')


@ambient
def rnn(hidden: int):
    """Elman cell: h' = tanh(x Wx + h Wh + b), emitted as the output;
    state initializes to zeros shaped like the input."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            wx=0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            wh=0.4 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros(hidden))

    def init(ndef, param: Struct) -> jax.Array:
        return jnp.zeros_like(ndef.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input @ param.wx + state @ param.wh + param.b)
        return h, h

    return node_def(apply, init=init, param=param, name='rnn')


@ambient
def moe(hidden: int, experts: int):
    """Soft mixture-of-experts with an internal residual, written over a
    (B, hidden) batch: the load-balance statistic is a population
    quantity. Emits that statistic (experts * sum(mean_gate^2); 1.0 =
    uniform) and per-expert usage as AUX — the tuple convention; the
    loss decides what to do with it."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            router=0.2 * jax.random.normal(rng.next(), (hidden, experts)),
            w=0.5 * jax.random.normal(rng.next(), (experts, hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros((experts, hidden)))

    def apply(param: Struct, input: jax.Array) -> tuple[jax.Array, Struct]:
        gates = jax.nn.softmax(input @ param.router, axis=-1)              # (B, E)
        expert_out = jnp.tanh(jnp.einsum('bh,ehk->bek', input, param.w) + param.b)
        mixed = jnp.einsum('be,beh->bh', gates, expert_out)
        usage = jnp.mean(gates, axis=0)                                    # (E,)
        balance = experts * jnp.sum(usage ** 2)
        return input + mixed, Struct(balance=balance, usage=usage)

    return node_def(apply, param=param, name='moe')
