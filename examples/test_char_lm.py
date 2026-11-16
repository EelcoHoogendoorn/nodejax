"""Character-level language modeling and autoregressive sampling on a 4.4 KB English corpus.

`build` composes an embedding table, a residual RNN stack, a mixture-of-experts block, and an
output projection inside an `ambient` scope supplying `vocab`, `hidden`, and `experts`, then ties
input and output weights with `tie(pipe, 'embed', 'unembed')` so the embedding matrix is stored
once at `param.embed` and accumulates gradients from both sites.

During unrolled sequence training under `scanned`, the MoE layer emits load-balancing statistics via
`Aux`, which `train_step` forwards to `lm_loss(..., *, aux)` alongside token cross-entropy.
`Sampler` closes the autoregressive loop as a two-member `Composite` pairing the language model
with a one-step `Delay` register and drawing categorical samples from the per-step `rng` stream.
"""

from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import (Aux, PNode, node, trained, Node, Composite, serial, stack,
                     scanned, residual, train_step, tie, split_aux,
                     ambient, nn)
from nodejax.control import Delay

HIDDEN, LAYERS, EXPERTS = 48, 3, 4
T, BATCH, STEPS = 16, 8, 3


def corpus() -> tuple[jax.Array, list[str]]:
    """Loads the bundled corpus as integer character IDs and the sorted vocabulary list."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'tiny_corpus.txt')
    text = open(path).read()
    chars = sorted(set(text))
    ids = jnp.asarray([chars.index(c) for c in text], dtype=jnp.int32)
    return ids, chars


def build(vocab: int) -> Node:
    """Builds a character language model with tied input and output embeddings over an `ambient` scope."""
    with ambient(vocab=vocab, hidden=HIDDEN, experts=EXPERTS):
        pipe = serial(
            embed=nn.Embed(),
            core=stack(residual(nn.RNN()), n=LAYERS),
            moe=nn.MoE(),
            unembed=nn.Unembed(),
        )
    return tie(pipe, 'embed', 'unembed')


def lm_loss(logits: jax.Array, target: jax.Array, *, aux: Aux) -> jax.Array:
    """Sums token cross-entropy with the scaled MoE load-balancing auxiliary penalty."""
    ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))
    return ce + 0.01 * jnp.mean(aux.moe.balance)


@node
def Sampler(lm: Node | PNode, temperature: float = 0.8) -> Node:
    """Samples one autoregressive token per step by feeding the previous token from `Delay` into `lm`.

    Composite wiring routes auxiliary MoE outputs away from `logits` and supplies a per-step `rng`
    stream when unrolled with `scanned`.
    """
    def apply(self, tick: jax.Array, rng) -> jax.Array:
        logits = self.lm(self.last.state)
        char = jax.random.categorical(rng.next(), logits / temperature, axis=-1)
        # `self.last` records the sampled character in the one-step delay register for the next tick.
        self.last(char)
        return char

    return Composite(lm=lm, last=Delay().with_input(jnp.zeros((1,), dtype=jnp.int32)))(apply, name=f'sample({lm.name})')


def test_tied_assembly() -> None:
    """Verifies that `tie` stores the shared embedding weight once at `param.embed` and omits `param.unembed`."""
    ids, chars = corpus()
    lm = build(len(chars))
    model = lm.with_input(ids[:BATCH]).parameterize(
        rng=jax.random.PRNGKey(0))

    assert model.param.embed.weight.shape == (len(chars), HIDDEN)
    assert 'unembed' not in model.param

    _, out = model.initialize()(ids[:BATCH])
    logits, aux = split_aux(out)
    assert logits.shape == (BATCH, len(chars))
    assert aux.moe.usage.shape == (EXPERTS,)


def test_trains_generates() -> None:
    """Trains the tied character language model with MoE load balancing and verifies keyed generation."""
    ids, chars = corpus()
    vocab = len(chars)
    lm = build(vocab)

    # `windows` indexes `(STEPS, T, BATCH)` character slices paired with `windows + 1` targets.
    rs = np.random.RandomState(0)
    starts = rs.randint(0, len(ids) - T - 1, size=(STEPS, BATCH))
    offsets = np.arange(T)[None, :, None]
    windows = starts[:, None, :] + offsets

    # `scanned(lm)` unrolls `lm` across `(T, BATCH)` token sequences inside `train_step`.
    initial = scanned(lm).with_input(ids[windows[0]]).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(initial, lm_loss, optax.adam(3e-3))
    final, aux = trained(trainer).apply(input=ids[windows], target=ids[windows + 1])

    assert aux.loss.shape == (STEPS,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, initial.param))

    # `final.param` matches the unwrapped `lm` parameter tree for single-batch inspection.
    fitted = lm.bind(final.param)
    _, (step_logits, report) = fitted.initialize(input=ids[:BATCH])(ids[:BATCH])
    assert step_logits.shape == (BATCH, vocab)
    assert report.moe.usage.shape == (EXPERTS,)
    assert jnp.all(jnp.isfinite(report.moe.usage))

    # `Sampler(fitted)` embeds the bound language model so `parameterize()` needs no PRNG key.
    gen = scanned(Sampler(fitted)).parameterize()
    ticks = jnp.zeros(16)
    sample_a, _ = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_b, _ = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_c, _ = gen.apply(rng=jax.random.PRNGKey(8), tick=ticks)

    assert jnp.all(sample_a == sample_b)
    assert not jnp.all(sample_a == sample_c)
    assert jnp.all((sample_a >= 0) & (sample_a < vocab))
