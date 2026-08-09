"""Window-size sweep: input history against model memory.

Four rows across lag-window sizes, same family, same budget, scored on
the same stream-time targets (t >= SUPPORT) whatever the window:

    mlp       fixed memoryless: only the window carries history
    rnn       fixed recurrent: hidden state carries history
    ttt-mlp   adapting memoryless: weights may carry history
    ttt-rnn   adapting recurrent: both memories at once

The window-one column is the question the sweep exists for: a FIXED
memoryless model is capped at the conditional mean there (the map from
one sample to the next is multivalued for a two-sine), while an
ADAPTING one may track the phase through its weights alone — whether
weight adaptation substitutes for input history, and at what window
size the rows converge.

Prints the mse table and writes plots/window_sweep.png.

MEASURED (single seed per cell, the shared budget):

            w=1       w=2       w=4       w=8       w=16
    mlp     0.18      0.09      0.13      0.18      0.25
    rnn     0.05      0.09      0.06      0.14      0.08
    ttt-mlp 0.18      0.07      0.04      0.012     0.018
    ttt-rnn 0.008     0.008     0.007     0.007     0.008

ttt-rnn is flat: with state AND weight adaptation the window is
redundant. ttt-mlp at window one trains stably yet gains nothing over
the fixed mlp: tracking phase through weights alone needs the weights
to swing at signal frequency, and the rate that would do it is the
rate the unsaturating inner diverges at, so at stability-compatible
rates weight memory cannot substitute for state at signal timescale —
it needs the window, and rides it down to near the ttt-rnn floor by
w=8. The one-dimensional learning signal itself is not the problem:
ttt-rnn thrives on the same signal at window one. Weight adaptation
complements state; it does not replace it.

Run directly:  python -m nodejax.examples.comparisons.ttt_window_sweep
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import scan, batch, train_step
from nodejax.struct import Struct

from nodejax.examples.comparisons.ttt_common import (
    HIDDEN, TASKS, META_STEPS, TTT_LR0, META_LR, make_tasks, query_scorer)
from nodejax.examples.comparisons.ttt_nodejax import RNN, model_ttt, feed_samples
from nodejax.examples.comparisons.ttt_variants import MLP, TTT_MLP_LR0, feed_lags

WINDOWS = (1, 2, 4, 8, 16)


def run_row(model, feed, lags: int) -> tuple[float, bool]:
    """Meta-train one row at one window size; return (held-out query
    mse, whether the run stayed finite)."""
    loss = query_scorer(lags)
    trainer = train_step(batch(model), loss, optax.adam(META_LR))
    m = batch(model).parameterize(rng=jax.random.PRNGKey(0))
    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS, lags)
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    final, losses = trainer.scan(trainer.init(model=m.param),
                                 Struct(input=jax.tree.map(fold, feed(train)),
                                        target=fold(train.target)))
    tasks = make_tasks(np.random.RandomState(99), TASKS, lags)
    out = batch(model).apply(final.model, feed(tasks))
    q = float(loss(out, tasks.target))
    return q, bool(jnp.all(jnp.isfinite(losses))) and np.isfinite(q)


def rows(lags: int) -> dict:
    """Per row: a builder over the inner rate, the feed, and the rates
    to try — the seed rate first, cooled stepwise when a run diverges
    (divergence is data: the rate that held is part of the result)."""
    return {
        'mlp': (lambda lr: batch(MLP(lags, HIDDEN)), feed_lags, (None,)),
        'rnn': (lambda lr: scan(RNN(lags, HIDDEN)), feed_lags, (None,)),
        'ttt-mlp': (lambda lr: model_ttt(MLP(lags, HIDDEN), lr), feed_samples,
                    (TTT_MLP_LR0, TTT_MLP_LR0 / 3, TTT_MLP_LR0 / 10)),
        'ttt-rnn': (lambda lr: model_ttt(RNN(lags, HIDDEN), lr), feed_samples,
                    (TTT_LR0, TTT_LR0 / 3, TTT_LR0 / 10)),
    }


def main() -> None:
    table: dict = {}
    for w in WINDOWS:
        for name, (build, feed, lrs) in rows(w).items():
            for lr in lrs:
                q, finite = run_row(build(lr), feed, w)
                if finite:
                    break
            table.setdefault(name, {})[w] = q if finite else float('nan')
            cooled = '' if lr == lrs[0] else f'  (cooled to lr {lr:g})'
            verdict = f'query mse {q:.4f}' if finite else 'DIVERGED at every rate'
            print(f'window {w:2d}  {name:8s}  {verdict}{cooled}', flush=True)

    print(f'\n{"":8s}' + ''.join(f'w={w:<8d}' for w in WINDOWS))
    for name, per_w in table.items():
        print(f'{name:8s}' + ''.join(f'{per_w[w]:<10.4f}' for w in WINDOWS))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, per_w in table.items():
        ax.plot(WINDOWS, [per_w[w] for w in WINDOWS], marker='o', label=name)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    ax.set_xlabel('lag window size')
    ax.set_ylabel('query mse')
    ax.legend()
    fig.suptitle('input history against model memory')
    fig.tight_layout()
    path = os.path.join(os.path.dirname(__file__), 'plots', 'window_sweep.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
