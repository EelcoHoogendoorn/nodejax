"""A character-level language model on a small vendored corpus (4.4KB of
public-domain English), end to end.

TIE — one embedding matrix serves both ends of the model. tie('embed',
'unembed') reparameterizes the pipe so the composite param carries the
matrix ONCE; the unembed slot is literally empty, and gradients from
both uses accumulate through the expansion.

AUX LOSS — the mixture-of-experts layer emits its load-balance statistic
on the aux channel: (output, aux) from the block, diverted under the
member's name by the pipe, stacked over time by scan, split by the LOSS
function and fed to the optimizer.

GENERATION — sampling is a feedback loop: a cyclic node whose state
carries (model state, last char, rng); each step feeds the model its own
previous sample, the rng auto-advances, and scan rolls it out.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import NodeDef, node_def, serial, stack, scan, residual, train_step, tie, split_aux, ambient, nn
from nodejax.types import Param
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


def build(vocab: int) -> NodeDef:
    """embed >> residual rnn stack >> MoE >> unembed, embeddings TIED.
    Every block is stock nn; the shared design arguments flow through
    ambient scope, so the members stay plain reusable factories."""
    with ambient(vocab=vocab, hidden=HIDDEN, experts=EXPERTS):
        pipe = serial(
            embed=nn.embed(),
            core=stack(residual(nn.rnn()), n=LAYERS),
            moe=nn.moe(),
            unembed=nn.unembed(),
        )
    return tie(pipe, 'embed', 'unembed')


# --- loss: cross-entropy + the aux load-balance term ---

def lm_loss(pred: tuple, target: jax.Array) -> jax.Array:
    logits, aux = split_aux(pred)          # the aux channel arrives IN the loss
    ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))
    return ce + 0.01 * jnp.mean(aux.moe.balance)


# --- generation: sampling as the feedback loop ---

def sampler_def(lm: NodeDef, temperature: float = 0.8) -> NodeDef:
    """Autoregressive sampling as a feedback loop: state carries
    (model state, last char, rng); each step feeds the model its own
    previous sample; the rng auto-advances."""
    def init(param: Param, rng: jax.Array) -> Struct:
        start = jnp.zeros((1,), dtype=jnp.int32)
        return Struct(inner=lm.build_state(param, input=start), last=start, rng=rng)

    def apply(param: Param, state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        new_inner, out = lm.apply_fn(param, state.inner, state.last)
        logits, _ = split_aux(out)
        char = jax.random.categorical(state.rng, logits / temperature, axis=-1)
        return Struct(inner=new_inner, last=char, rng=state.rng), char

    lifted = node_def(apply, init=init, param=lambda: (), name=f'sample({lm.name})')
    return lifted._replace(param_fn=lm._param_impl, param_input_spec=lm.param_input_spec,
                           param_reads_shape=lm.param_reads_shape)


# --- tests ---

def test_tied_assembly():
    """The tied pipe carries the embedding ONCE: unembed's param slot is
    empty, and the model works end to end through both uses."""
    ids, chars = corpus()
    lm = build(len(chars))
    model = lm.parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.embed.weight.shape == (len(chars), HIDDEN)
    assert len(jax.tree.leaves(model.param.unembed)) == 0        # tied away

    state = model.with_input(ids[:BATCH]).init()
    new_state, out = model.apply(state, ids[:BATCH])
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
    model = lm.parameterize(rng=jax.random.PRNGKey(0))
    rollout = scan(lm)                       # (T, B) ids -> per-step outputs

    # windows of the corpus: input chars and next-char targets
    rs = np.random.RandomState(0)
    starts = rs.randint(0, len(ids) - T - 1, size=(STEPS, BATCH))
    offsets = np.arange(T)[None, :, None]                       # (1, T, 1)
    windows = starts[:, None, :] + offsets                      # (S, T, B)
    stream = Struct(input=ids[windows], target=ids[windows + 1])

    trainer = train_step(rollout.bind(model.param), lm_loss, optax.adam(3e-3))
    final, losses = trainer.scan(trainer.init(model=model.param), stream)

    uniform = jnp.log(vocab)
    # untrained logits carry the init scale, so CE starts at or above the
    # uniform floor; training must fall far below that floor
    assert losses[0] > 0.9 * uniform
    assert losses[-1] < 0.7 * uniform, losses[-1]               # learned real structure

    # the aux loss did its job: no expert collapsed
    _, out = lm.apply(final.model, lm.init(final.model, input=ids[:BATCH]), ids[:BATCH])
    _, aux = split_aux(out)
    assert jnp.max(aux.moe.usage) < 0.7, aux.moe.usage

    # generation: the feedback loop, keyed through the reserved rng input
    sampler = sampler_def(lm)
    gen = scan(sampler).bind(final.model)
    ticks = jnp.zeros(160)
    sample_a = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_b = gen.apply(rng=jax.random.PRNGKey(7), tick=ticks)
    sample_c = gen.apply(rng=jax.random.PRNGKey(8), tick=ticks)

    assert jnp.all(sample_a == sample_b)                        # keyed determinism
    assert not jnp.all(sample_a == sample_c)
    assert jnp.all((sample_a >= 0) & (sample_a < vocab))

    text = ''.join(chars[int(c)] for c in sample_a[:, 0])
    print(f"\n[char-lm] vocab {vocab} | loss {losses[0]:.2f} -> {losses[-1]:.2f} "
          f"(uniform {uniform:.2f}) | expert usage {np.round(np.asarray(aux.moe.usage), 2)}"
          f"\n[sample] {text!r}")
