"""Adaptation mechanisms compared on one sequence-prediction family.

THE TASK FAMILY: each task is a stream x_t = a1 sin(2 pi f1 t + p1) +
a2 sin(2 pi f2 t + p2) + noise — amplitudes, frequencies and phases
drawn per task, features O(1) throughout. The model sees windows and
predicts the next sample. A short window under observation noise
under-determines the four-parameter task, so a fixed predictor is
capped, and identifying the task from history is what adaptation is
for. Every task's stream splits into a SUPPORT region (available for
adaptation) and a QUERY region (scored); every row meta-trains with
the same outer budget on the same meta-objective: query-region
next-step error.

THE ROWS — same data, same scoring, different adaptation:

    mlp         one fixed memoryless predictor for the whole family
    rnn         one fixed recurrent net: no weight adaptation, but
                the hidden state identifies the task in-context —
                the baseline for the weight-adapting rows
    ttt-linear  streaming: scan(ttt(linear)) — the predictor's
                weights adapt every step on the self-supervised
                next-step loss, through support into query
    ttt-mlp     streaming, two-layer gelu inner (the RoboTTT inner
                model)
    ttt-rnn     streaming, recurrent inner: weights adapt by gradient
                step AND the cell carries hidden state — two memories
                at two speeds in one cell
    ttt-ssm     streaming, diagonal state-space inner: sigmoid-
                bounded poles, linear signal path — the stability
                contrast to ttt-rnn
    ttt-mingru2 streaming, two stacked min-gru layers under one ttt:
                gated cells without a hidden-to-hidden matrix —
                weaker per layer, stacked deeper to compensate — with
                tanh candidates, so the saturation argument predicts
                the hot rate holds at depth

The rnn/ssm pair measures where ttt stability lives. The tanh cell's
state is confined to [-1, 1] whatever the inner loop does to its
weights: it runs at the hot rate and improves with stream length.
The ssm's bounded poles protect only the recurrence — its signal
path is linear end to end, so nothing bounds amplitude under
per-step weight updates: it needs the cooler rate at this stream
length and diverges on longer ones. Under weight adaptation, safety
lives in the signal path's saturation, not the recurrence's pole
constraints.

Episodic adaptation (metasgd) is the meta-controller example's
mechanism, at home where support episodes are physically discrete
test runs; the mechanisms compared here are the ones native to a
single stream — memory against weight adaptation.

Every row consumes the same train_step-style stream:
Struct(input=<L lags>, target=<the next value>). For the ttt rows
this IS the self-supervision — the target column is derived from the
stream itself, and ttt's predict-then-update order keeps scoring
prequential (every emitted prediction comes from weights that have not
trained on its target). Supervised rows read the same columns as
labeled data.

Writes one figure per row to plots/meta_<name>.png and a summary
table to stdout.

Run directly:  python -m nodejax.examples.ttt_nodejax
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import (NodeDef, node_def, scan, batch, stack, ttt,
                           train_step, KeyStream)
from nodejax.struct import Struct

LAGS = 8
STREAM, SUPPORT = 192, 128         # steps per task; support/query boundary
HIDDEN = 16
TASKS, META_STEPS = 8, 400
TTT_LR0, META_LR = 0.01, 1e-3
TTT_MLP_LR0 = 0.003  # the gelu inner diverges at TTT_LR0: unlike the tanh
                     # cells, nothing in it saturates, so nothing bounds the
                     # per-step gradient — it gets the cooler seed rate
TTT_SSM_LR0 = 0.003  # the ssm inner shares the disease: poles are bounded
                     # but the signal path is linear, so amplitude is not
NOISE = 0.05


# --- the task family ---

def make_tasks(rs: np.random.RandomState, n_tasks: int):
    """Per task: the stream's sample vectors and next-step targets.

    Returns Struct(input=(n, M, LAGS), target=(n, M)) with M sample
    positions: input[i] holds x[i .. i+LAGS-1], target[i] = x[i+LAGS]
    — the value right after the lags, which every row forecasts.
    Position i < QUERY0 lies in the support region."""
    t = np.arange(STREAM)[None, :]
    def draw(lo, hi):
        return rs.uniform(lo, hi, (n_tasks, 1))
    x = (draw(0.5, 1.5) * np.sin(2 * np.pi * draw(0.02, 0.1) * t + draw(0, 2 * np.pi))
         + draw(0.2, 0.8) * np.sin(2 * np.pi * draw(0.1, 0.25) * t + draw(0, 2 * np.pi))
         + NOISE * rs.standard_normal((n_tasks, STREAM))).astype(np.float32)
    M = STREAM - LAGS
    lags = np.stack([x[:, i:i + LAGS] for i in range(M)], axis=1)
    return Struct(input=jnp.asarray(lags), target=jnp.asarray(x[:, LAGS:]))


# The support/query boundary, converted to position coordinates. A
# stream has two coordinate systems: stream time t (STREAM values)
# and sample position i (M = STREAM - LAGS samples), where sample i
# reads lags x[i .. i+LAGS-1] and targets x[i+LAGS]. The boundary is
# defined in stream time — targets at t >= SUPPORT are query — so the
# first query position is the i whose target lands exactly there:
# i + LAGS = SUPPORT.
QUERY0 = SUPPORT - LAGS


# --- predictors: lags in, one forecast out ---

def linear_def() -> NodeDef:
    def param(rng: KeyStream) -> Struct:
        return Struct(w=0.1 * jax.random.normal(rng.next(), (LAGS,)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.w @ input

    return node_def(apply, param=param, name='lin')


def mlp_def(hidden: int) -> NodeDef:
    def param(rng: KeyStream) -> Struct:
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (LAGS, hidden)),
                      b1=jnp.zeros(hidden),
                      w2=0.3 * jax.random.normal(rng.next(), (hidden,)),
                      b2=jnp.zeros(()))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.w2 @ jax.nn.gelu(input @ param.w1 + param.b1) + param.b2

    return node_def(apply, param=param, name='mlp')


def rnn_def(hidden: int) -> NodeDef:
    """Recurrent predictor: the hidden state advances on the lags —
    each stream value enters the recurrence as it ages into one."""
    def param(rng: KeyStream) -> Struct:
        return Struct(win=0.5 * jax.random.normal(rng.next(), (LAGS, hidden)),
                      wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
                      b=jnp.zeros(hidden),
                      wout=0.3 * jax.random.normal(rng.next(), (hidden,)))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.b.shape)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input @ param.win + param.wh @ state + param.b)
        return h, param.wout @ h

    return node_def(apply, init=init, param=param, name='rnn')


def ssm_def(hidden: int) -> NodeDef:
    """Diagonal state-space predictor: per-unit poles sigmoid-bounded
    in (0, 1) — the recurrence stays stable under any weight update,
    including updates to the poles themselves. The safety it cannot
    value is amplitude: input and readout are linear, so the state
    scales freely with the adapted weights (see TTT_SSM_LR0)."""
    def param(rng: KeyStream) -> Struct:
        return Struct(p=jax.random.uniform(rng.next(), (hidden,), minval=0.5, maxval=3.0),
                      win=0.5 * jax.random.normal(rng.next(), (LAGS, hidden)),
                      wout=0.3 * jax.random.normal(rng.next(), (hidden,)))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.p.shape)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        a = jax.nn.sigmoid(param.p)
        h = a * state + (1 - a) * (input @ param.win)
        return h, param.wout @ h

    return node_def(apply, init=init, param=param, name='ssm')


def mingru_def(hidden: int) -> NodeDef:
    """One min-gru layer, width-preserving: gate and candidate depend
    on the input alone, the state is a gated convex mix — bounded, the
    tanh candidate saturating."""
    def param(rng: KeyStream) -> Struct:
        def w() -> jax.Array:
            return 0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden)
        return Struct(wz=w(), bz=jnp.zeros(hidden), wh=w(), bh=jnp.zeros(hidden))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.bz.shape)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = jax.nn.sigmoid(input @ param.wz + param.bz)
        hc = jnp.tanh(input @ param.wh + param.bh)
        h = (1 - z) * state + z * hc
        return h, h

    return node_def(apply, init=init, param=param, name='mingru')


def mingru_pred(hidden: int, layers: int) -> NodeDef:
    """Projection >> stacked min-gru cells >> scalar head, wrapped by
    ttt as ONE node: the whole pipe's params become the adapted
    state."""
    def pparam(rng: KeyStream) -> Struct:
        return Struct(w=0.5 * jax.random.normal(rng.next(), (LAGS, hidden)))

    def hparam(rng: KeyStream) -> Struct:
        return Struct(w=0.3 * jax.random.normal(rng.next(), (hidden,)))

    proj = node_def(lambda param, input: input @ param.w, param=pparam, name='proj')
    head = node_def(lambda param, input: param.w @ input, param=hparam, name='head')
    return proj >> stack(mingru_def(hidden), n=layers) >> head


def runstats_def(momentum: float) -> NodeDef:
    """Running standardizer over the lag vector: per-feature EMA mean
    and variance as cyclic state, updated every step by apply itself.
    Under ttt this state rides the wrapper's `inner` slot untouched —
    updated by the node, carried by the scan, invisible to the
    gradient step, which adapts params alone."""
    def init(ndef) -> Struct:
        return Struct(m1=jnp.zeros_like(ndef.input), var=jnp.ones_like(ndef.input))

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        out = (input - state.m1) / jnp.sqrt(state.var + 1e-5)
        new = Struct(m1=(1 - momentum) * state.m1 + momentum * input,
                     var=(1 - momentum) * state.var + momentum * (input - state.m1) ** 2)
        return new, out

    return node_def(apply, init=init, name='runstats')


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def query_mse(out, target):
    """The shared meta-objective. Both arrays carry sample positions
    on their LAST axis — one forecast and one target per position —
    with leading axes being whatever batching added (tasks, and in
    meta-training the meta-step). The slice keeps the positions whose
    targets lie past the support boundary, so every row is scored on
    the query region alone, whatever it did with the support region:
    the ttt rows streamed through it adapting, the baselines carried
    at most their hidden state out of it."""
    return jnp.mean((out[..., QUERY0:] - target[..., QUERY0:]) ** 2)


# --- one model per row: forecasts (M,) per task; the ttt rows
# consume the sample stream whole, the rest just the lags ---

def model_ttt(predictor: NodeDef, lr0: float) -> NodeDef:
    return scan(ttt(predictor, mse, lr0))                # weights adapt down the stream


def feed_lags(tasks):
    """The supervised rows' model input: the lags alone."""
    return tasks.input


def feed_samples(tasks):
    """The ttt rows' model input: the task struct as-is — make_tasks
    already emits train_step's sample stream, targets included,
    because the ttt cell trains as it goes."""
    return tasks


# --- run one row: meta-train, evaluate, plot ---

def report(name: str, n_weights: int, finite: bool, out, tasks) -> None:
    q = query_mse(out, tasks.target)
    plot(name, out, tasks)
    print(f'{name:12s} weights={n_weights:4d}: finite={finite} '
          f'query mse {q:.4f}', flush=True)


def plot(name: str, out, tasks) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    for i, ax in enumerate(axs.flat):
        ax.axvspan(QUERY0, tasks.target.shape[1], color='k', alpha=0.07, label='query')
        ax.plot(tasks.target[i], 'k:', lw=1.5, label='next sample')
        ax.plot(np.asarray(out)[i], 'C1', lw=1, label='predicted')
        ax.set_ylim(-2.5, 2.5)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(name)
    fig.tight_layout()
    path = os.path.join(os.path.dirname(__file__), 'plots', f'meta_{name}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stream(name: str, model: NodeDef, feed) -> None:
    """Rows without an episodic split: none and the ttt family. feed
    picks the model's input off the task struct."""
    trainer = train_step(batch(model), query_mse, optax.adam(META_LR))
    m = batch(model).parameterize(rng=jax.random.PRNGKey(0))
    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    final, losses = trainer.scan(trainer.init(model=m.param), Struct(input=jax.tree.map(fold, feed(train)), target=fold(train.target)))

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    out = batch(model).apply(final.model, feed(tasks))
    n = sum(x.size for x in jax.tree.leaves(final.model))
    report(name, n, bool(jnp.all(jnp.isfinite(losses))), out, tasks)


def main() -> None:
    run_stream('mlp', batch(mlp_def(HIDDEN)), feed_lags)
    run_stream('rnn', scan(rnn_def(HIDDEN)), feed_lags)
    run_stream('ttt-linear', model_ttt(linear_def(), TTT_LR0), feed_samples)
    run_stream('ttt-mlp', model_ttt(mlp_def(HIDDEN), TTT_MLP_LR0), feed_samples)
    run_stream('ttt-rnn', model_ttt(rnn_def(HIDDEN), TTT_LR0), feed_samples)
    run_stream('ttt-ssm', model_ttt(ssm_def(HIDDEN), TTT_SSM_LR0), feed_samples)
    run_stream('ttt-stats-rnn',
               model_ttt(runstats_def(0.05) >> rnn_def(HIDDEN), TTT_LR0), feed_samples)
    run_stream('ttt-mingru2', model_ttt(mingru_pred(HIDDEN, 2), TTT_LR0), feed_samples)


if __name__ == '__main__':
    main()
