"""A character-level language model on a small vendored corpus (4.4KB of
public-domain English), end to end.

TIE — one embedding matrix serves both ends of the model. tie('embed',
'unembed') reparameterizes the pipe so the composite param carries the
matrix ONCE; the unembed slot is literally empty, and gradients from
both uses accumulate through the expansion.

AUX LOSS — the mixture-of-experts layer emits its load-balance statistic
on the aux stream: (output, aux) from the block, diverted under the
member's name by the pipe, stacked over time by scan, and supplied to the
loss through its declared ``aux`` argument.

GENERATION — sampling is a feedback loop: a cyclic node whose state
carries (model state, last char, rng); each step feeds the model its own
previous sample, the rng auto-advances, and scan rolls it out.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import (Aux, node, trained, Node, Leaf, Composite, serial, stack,
                     scan, scanned, residual, train_step, tie, split_aux,
                     ambient, nn)
from nodejax.control import Delay
from nodejax.core.types import Param
from nodejax.struct import Struct

HIDDEN, LAYERS, EXPERTS = 48, 3, 4
T, BATCH, STEPS = 32, 32, 1500


# --- corpus ---

def corpus() -> tuple[jax.Array, list[str]]:
    path = os.path.join(os.path.dirname(__file__), 'data', 'tiny_corpus.txt')
    text = open(path).read()
    chars = sorted(set(text))
    ids = jnp.asarray([chars.index(c) for c in text], dtype=jnp.int32)
    return ids, chars


def build(vocab: int) -> Node:
    """embed >> residual rnn stack >> MoE >> unembed, embeddings TIED.
    Every block is stock nn; the shared design arguments flow through
    ambient scope, so the members stay plain reusable factories."""
    with ambient(vocab=vocab, hidden=HIDDEN, experts=EXPERTS):
        pipe = serial(
            embed=nn.Embed(),
            core=stack(residual(nn.RNN()), n=LAYERS),
            moe=nn.MoE(),
            unembed=nn.Unembed(),
        )
    return tie(pipe, 'embed', 'unembed')


# --- loss: cross-entropy + the aux load-balance term ---

def lm_loss(logits: jax.Array, target: jax.Array, *, aux: Aux) -> jax.Array:
    ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))
    return ce + 0.01 * jnp.mean(aux.moe.balance)


# --- generation: sampling as the feedback loop ---

@node
def Sampler(lm: Node, temperature: float = 0.8) -> Node:
    """Autoregressive sampling, spelled as what it is: the model and the
    register it feeds itself through.

    Each step reads the character emitted last, runs the model on it, and
    draws the next. The register is the stock one-tick Delay, because that
    is exactly what it is. The draw itself is arithmetic on logits and a
    key, so it stays a function: a node owning neither params nor state
    would add a member slot for nothing.

    Nothing here unwraps an aux stream or threads a key. The wiring
    diverts the model's sown losses and hands this apply clean logits, and
    the boundary key arrives per STEP — scan splits it across time exactly
    as a composite splits it across members."""
    def apply(self, tick: jax.Array, rng) -> jax.Array:
        logits = self.lm(self.state.last)
        char = jax.random.categorical(rng.next(), logits / temperature, axis=-1)
        self.last(char)                        # store it for the next step
        return char

    return Composite(lm=lm, last=Delay().with_input(jnp.zeros((1,), dtype=jnp.int32)))(apply, name=f'sample({lm.name})')


# --- tests ---

def test_tied_assembly():
    """The tied pipe carries the embedding ONCE: unembed's param slot is
    empty, and the model works end to end through both uses."""
    ids, chars = corpus()
    lm = build(len(chars))
    model = lm.with_input(ids[:BATCH]).parameterize(
        rng=jax.random.PRNGKey(0))

    assert model.param.embed.weight.shape == (len(chars), HIDDEN)
    assert 'unembed' not in model.param                          # tied away

    _, out = model.initialize()(ids[:BATCH])
    logits, aux = split_aux(out)
    assert logits.shape == (BATCH, len(chars))
    assert aux.moe.usage.shape == (EXPERTS,)


def test_trains_generates():
    """End to end on real text: loss falls far below the uniform floor,
    the aux term keeps the experts from collapsing, and the trained model
    GENERATES text through the feedback sampler."""
    ids, chars = corpus()
    vocab = len(chars)
    lm = build(vocab)

    # windows of the corpus: input chars and next-char targets
    rs = np.random.RandomState(0)
    starts = rs.randint(0, len(ids) - T - 1, size=(STEPS, BATCH))
    offsets = np.arange(T)[None, :, None]                       # (1, T, 1)
    windows = starts[:, None, :] + offsets                      # (S, T, B)

    # the rollout eats (T, B) ids and emits per-step outputs
    trainer = train_step(
        scanned(lm).with_input(ids[windows[0]]).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        lm_loss, optax.adam(3e-3))
    final, aux = trained(trainer).apply(input=ids[windows], target=ids[windows + 1])

    uniform = jnp.log(vocab)
    # untrained logits carry the init scale, so CE starts at or above the
    # uniform floor; training must fall far below that floor
    assert aux.loss[0] > 0.9 * uniform
    assert aux.loss[-1] < 0.7 * uniform, aux.loss[-1]               # learned real structure

    # the aux loss did its job: no expert collapsed. `final` is the
    # trained ROLLOUT, which eats a (T, B) sequence; this check wants one
    # batch through the lm itself, and a wrapper's param IS the lm's, so
    # binding the lm to it is exact rather than lucky
    fitted = lm.bind(final.param)
    _, (_, report) = fitted.initialize(input=ids[:BATCH])(ids[:BATCH])
    assert jnp.max(report.moe.usage) < 0.7, report.moe.usage

    # generation: the feedback loop, keyed through the explicit RNG channel.
    # The TRAINED lm composes in as a bound member (the transport
    # container), and the bare parameterize fills every slot from that
    # storage, owing no key. The rollout emits (characters, aux); the
    # MoE's load-balance term remains outside the sampler's primary output
    gen = scanned(Sampler(fitted)).parameterize()
    ticks = jnp.zeros(160)
    sample_a, _ = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_b, _ = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_c, _ = gen.apply(rng=jax.random.PRNGKey(8), tick=ticks)

    assert jnp.all(sample_a == sample_b)                        # keyed determinism
    assert not jnp.all(sample_a == sample_c)
    assert jnp.all((sample_a >= 0) & (sample_a < vocab))

    text = ''.join(chars[int(c)] for c in sample_a[:, 0])
    print(f"\n[char-lm] vocab {vocab} | loss {aux.loss[0]:.2f} -> {aux.loss[-1]:.2f} "
          f"(uniform {uniform:.2f}) | expert usage {np.round(np.asarray(report.moe.usage), 2)}"
          f"\n[sample] {text!r}")
