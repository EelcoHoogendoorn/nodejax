"""Adaptation mechanisms compared on one sequence-prediction family.

The framework comparison lives in ttt_nodejax.py (one row, four
implementations); this file is the wider study on the same harness —
same data, same scoring, different adaptation:

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
    ttt-stats-rnn  the rnn inner behind a running standardizer, whose
                stats ride the state slot untouched by the gradient
                step
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

Writes plots/meta_<name>.png per row and a summary table to stdout.

Run directly:  python -m nodejax.examples.comparisons.ttt_variants
"""

import jax
import jax.numpy as jnp

from nodejax import NodeDef, node_def, scan, batch, stack, KeyStream
from nodejax.struct import Struct

from nodejax.examples.comparisons.ttt_common import LAGS, HIDDEN, TTT_LR0
from nodejax.examples.comparisons.ttt_nodejax import (
    RNN, model_ttt, feed_samples, run_stream)

TTT_MLP_LR0 = 0.003  # the gelu inner diverges at TTT_LR0: unlike the tanh
                     # cells, nothing in it saturates, so nothing bounds the
                     # per-step gradient — it gets the cooler seed rate
TTT_SSM_LR0 = 0.003  # the ssm inner shares the disease: poles are bounded
                     # but the signal path is linear, so amplitude is not


# --- predictors: lags in, one forecast out ---

def Linear(width: int) -> NodeDef:
    def param(rng: KeyStream) -> Struct:
        return Struct(w=0.1 * jax.random.normal(rng.next(), (width,)))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.w @ input

    return node_def(apply, param=param, name='lin')


def MLP(width: int, hidden: int) -> NodeDef:
    def param(rng: KeyStream) -> Struct:
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (width, hidden)),
                      b1=jnp.zeros(hidden),
                      w2=0.3 * jax.random.normal(rng.next(), (hidden,)),
                      b2=jnp.zeros(()))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.w2 @ jax.nn.gelu(input @ param.w1 + param.b1) + param.b2

    return node_def(apply, param=param, name='mlp')


def SSM(width: int, hidden: int) -> NodeDef:
    """Diagonal state-space predictor: per-unit poles sigmoid-bounded
    in (0, 1) — the recurrence stays stable under any weight update,
    including updates to the poles themselves. The safety it cannot
    value is amplitude: input and readout are linear, so the state
    scales freely with the adapted weights (see TTT_SSM_LR0)."""
    def param(rng: KeyStream) -> Struct:
        return Struct(p=jax.random.uniform(rng.next(), (hidden,), minval=0.5, maxval=3.0),
                      win=0.5 * jax.random.normal(rng.next(), (width, hidden)),
                      wout=0.3 * jax.random.normal(rng.next(), (hidden,)))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.p.shape)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        a = jax.nn.sigmoid(param.p)
        h = a * state + (1 - a) * (input @ param.win)
        return h, param.wout @ h

    return node_def(apply, init=init, param=param, name='ssm')


def MinGRU(hidden: int) -> NodeDef:
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


def mingru_pred(width: int, hidden: int, layers: int) -> NodeDef:
    """Projection >> stacked min-gru cells >> scalar head, wrapped by
    ttt as ONE node: the whole pipe's params become the adapted
    state."""
    def pparam(rng: KeyStream) -> Struct:
        return Struct(w=0.5 * jax.random.normal(rng.next(), (width, hidden)))

    def hparam(rng: KeyStream) -> Struct:
        return Struct(w=0.3 * jax.random.normal(rng.next(), (hidden,)))

    proj = node_def(lambda param, input: input @ param.w, param=pparam, name='proj')
    head = node_def(lambda param, input: param.w @ input, param=hparam, name='head')
    return proj >> stack(MinGRU(hidden), n=layers) >> head


def RunStats(momentum: float) -> NodeDef:
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


def feed_lags(tasks):
    """The supervised rows' model input: the lags alone."""
    return tasks.input


def main() -> None:
    run_stream('mlp', batch(MLP(LAGS, HIDDEN)), feed_lags)
    run_stream('rnn', scan(RNN(LAGS, HIDDEN)), feed_lags)
    run_stream('ttt-linear', model_ttt(Linear(LAGS), TTT_LR0), feed_samples)
    run_stream('ttt-mlp', model_ttt(MLP(LAGS, HIDDEN), TTT_MLP_LR0), feed_samples)
    run_stream('ttt-rnn', model_ttt(RNN(LAGS, HIDDEN), TTT_LR0), feed_samples)
    run_stream('ttt-ssm', model_ttt(SSM(LAGS, HIDDEN), TTT_SSM_LR0), feed_samples)
    run_stream('ttt-stats-rnn',
               model_ttt(RunStats(0.05) >> RNN(LAGS, HIDDEN), TTT_LR0), feed_samples)
    run_stream('ttt-mingru2', model_ttt(mingru_pred(LAGS, HIDDEN, 2), TTT_LR0), feed_samples)


if __name__ == '__main__':
    main()
