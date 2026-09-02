"""Shared task family, constants, scoring and reporting for the ttt
comparisons: NEXT-TOKEN prediction on synthetic Markov sequences.

THE TASK FAMILY: each task draws its own transition matrix over VOCAB
tokens (concentrated rows, so the chain has structure worth learning)
and samples one sequence from it. A fixed predictor is capped at the
family average; identifying THIS task's chain from the tokens read so
far is what adaptation is for. Targets past the support boundary are
scored; every row meta-trains with the same outer budget on the same
meta-objective, query-region next-token cross-entropy.

THE COMPARISON IS STRUCTURAL. The scores are sanity checks, not
results: the rows exist to compare how each
framework spells the same program. One structural row is the PAIRING
itself. The nodejax file feeds the raw token sequence twice, as input
and as target, and a next_step register pairs it in-graph; every
rival threads the previous token through its scan by hand.

The framework rival files carry their own line-identical copy of the
generator on purpose: each stays runnable as one self-contained file
in an environment without this package.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

from nodejax import split_aux

VOCAB, HIDDEN = 8, 16
STREAM, SUPPORT = 192, 128         # tokens per task; support/query boundary
TASKS, META_STEPS = 8, 300
TTT_LR0, META_LR = 0.05, 1e-3
CONCENTRATION = 2.0                # transition-logit scale
QUERY0 = SUPPORT                   # targets at t >= SUPPORT are query


def make_tasks(rs: np.random.RandomState, n_tasks: int):
    """Per task: one token sequence sampled from the task's own chain.
    Returns int32 tokens of shape (n_tasks, STREAM)."""
    logits = CONCENTRATION * rs.standard_normal((n_tasks, VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros((n_tasks, STREAM), dtype=np.int64)
    state = rs.randint(VOCAB, size=n_tasks)
    rows = np.arange(n_tasks)
    for t in range(STREAM):
        tokens[:, t] = state
        u = rs.random(n_tasks)[:, None]
        state = (P[rows, state].cumsum(-1) > u).argmax(-1)
    return jnp.asarray(tokens, dtype=jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    """Per-step next-token loss: logits (VOCAB,), target one token id."""
    return -jax.nn.log_softmax(logits)[target]


def sequence_nll(out: jax.Array, target: jax.Array) -> jax.Array:
    """Per-position negative log likelihood, nats: out carries logits
    with positions on the second-to-last axis, target the token ids."""
    logp = jax.nn.log_softmax(out)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return -picked


def query_xent(out: jax.Array, target: jax.Array) -> jax.Array:
    """The shared meta-objective: mean next-token cross-entropy over the
    query region. The adapting cell reports its own loss on the aux
    output; this objective scores the clean query window and nothing else.
    The first position is the register's primed pair and lies deep in
    the support region, so the slice never sees it."""
    return jnp.mean(sequence_nll(out, target)[..., QUERY0:])


def plot(name: str, nll) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nll = np.asarray(nll)
    fig, axs = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        ax.axvspan(QUERY0, nll.shape[-1], color='k', alpha=0.07, label='query')
        smooth = np.convolve(nll[i], np.ones(8) / 8, mode='valid')
        ax.plot(smooth, 'C1', lw=1, label='next-token nll (smoothed)')
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(name)
    fig.tight_layout()
    path = os.path.join(os.path.dirname(__file__), 'plots', f'meta_{name}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(name: str, n_weights: int, finite: bool, out: jax.Array, tokens: jax.Array) -> None:
    out, _ = split_aux(out)
    nll = sequence_nll(out, tokens)
    plot(name, np.asarray(nll).mean(0))
    print(f'{name:12s} weights={n_weights:4d}: finite={finite} '
          f'query xent {float(jnp.mean(nll[..., QUERY0:])):.2f}', flush=True)
