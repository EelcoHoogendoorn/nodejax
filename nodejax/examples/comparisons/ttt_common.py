"""Shared task family, constants, scoring and reporting for the ttt
comparisons.

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

THE LAG WINDOW IS NONSTANDARD, and exists only because the stream is
scalar. ttt in the literature adapts on the current observation alone,
one token or frame per step, and a rich observation carries enough
signal for that; a single scalar of a two-sine does not (the map from
one sample to the next is multivalued), so this harness fattens each
observation by stacking the last LAGS samples, and imposes that
interface on every row for comparability — including the rnn, which a
standard harness would feed one sample per step. The standard shape
would come from a richer task instead, a per-task readout of the
latent oscillators into a vector observed once per step;
ttt_window_sweep.py measures what the window buys each row in the
meantime.

The framework rival files (ttt_rnn_by_hand, ttt_rnn_flax,
ttt_rnn_torch) carry their own line-identical copy of this generator
on purpose: each stays runnable as one self-contained file in an
environment without this package.
"""

import os

import numpy as np
import jax.numpy as jnp

from nodejax.struct import Struct

LAGS = 8
STREAM, SUPPORT = 192, 128         # steps per task; support/query boundary
HIDDEN = 16
TASKS, META_STEPS = 8, 400
TTT_LR0, META_LR = 0.01, 1e-3
NOISE = 0.05

# The support/query boundary, converted to position coordinates. A
# stream has two coordinate systems: stream time t (STREAM values)
# and sample position i (M = STREAM - LAGS samples), where sample i
# reads lags x[i .. i+LAGS-1] and targets x[i+LAGS]. The boundary is
# defined in stream time — targets at t >= SUPPORT are query — so the
# first query position is the i whose target lands exactly there:
# i + LAGS = SUPPORT.
QUERY0 = SUPPORT - LAGS


def make_tasks(rs: np.random.RandomState, n_tasks: int, lags: int = LAGS):
    """Per task: the stream's sample vectors and next-step targets.

    Returns Struct(input=(n, M, lags), target=(n, M)) with M sample
    positions: input[i] holds x[i .. i+lags-1], target[i] = x[i+lags]
    — the value right after the lags, which every row forecasts.
    Position i < SUPPORT - lags lies in the support region."""
    t = np.arange(STREAM)[None, :]
    def draw(lo, hi):
        return rs.uniform(lo, hi, (n_tasks, 1))
    x = (draw(0.5, 1.5) * np.sin(2 * np.pi * draw(0.02, 0.1) * t + draw(0, 2 * np.pi))
         + draw(0.2, 0.8) * np.sin(2 * np.pi * draw(0.1, 0.25) * t + draw(0, 2 * np.pi))
         + NOISE * rs.standard_normal((n_tasks, STREAM))).astype(np.float32)
    M = STREAM - lags
    windows = np.stack([x[:, i:i + lags] for i in range(M)], axis=1)
    return Struct(input=jnp.asarray(windows), target=jnp.asarray(x[:, lags:]))


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def query_scorer(lags: int = LAGS):
    """query_mse for a chosen window size: the scored region is stream
    time t >= SUPPORT whatever the window, so rows built on different
    windows score the same targets."""
    q0 = SUPPORT - lags

    def loss(out, target):
        return jnp.mean((out[..., q0:] - target[..., q0:]) ** 2)

    return loss


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


def report(name: str, n_weights: int, finite: bool, out, tasks) -> None:
    q = query_mse(out, tasks.target)
    plot(name, out, tasks)
    print(f'{name:12s} weights={n_weights:4d}: finite={finite} '
          f'query mse {q:.4f}', flush=True)
