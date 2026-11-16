"""Shared task generator, constants, and verification checks for the TTT comparison.

Each task draws a task-specific Markov transition matrix over `VOCAB` tokens and samples a sequence of length `STREAM`. A predictor adapts online on each observed token before predicting the next token, and an outer Adam loop meta-learns both the initial weights and per-parameter inner learning rates to minimize cross-entropy over the query region (`t >= QUERY0`).
"""

import os

import numpy as np

SMOKE = os.environ.get('NODEJAX_SMOKE') == '1'
VOCAB, HIDDEN = 8, 16
STREAM, SUPPORT = (16, 8) if SMOKE else (192, 128)
TASKS, EVAL_TASKS, META_STEPS = (3, 4, 2) if SMOKE else (8, 32, 150)
TTT_LR0, META_LR = 0.05, 3e-3
CONCENTRATION = 2.0
QUERY0 = SUPPORT
EXPECTED_WEIGHTS = 2 * (VOCAB * HIDDEN + HIDDEN * HIDDEN + HIDDEN * VOCAB)


def make_tasks(rs: np.random.RandomState, n_tasks: int) -> np.ndarray:
    """Samples `n_tasks` token sequences of shape `(n_tasks, STREAM)` (`int32`)."""
    logits = CONCENTRATION * rs.standard_normal((n_tasks, VOCAB, VOCAB))
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    tokens = np.zeros((n_tasks, STREAM), dtype=np.int32)
    state = rs.randint(VOCAB, size=n_tasks)
    rows = np.arange(n_tasks)
    for t in range(STREAM):
        tokens[:, t] = state
        u = rs.random(n_tasks)[:, None]
        state = (probs[rows, state].cumsum(-1) > u).argmax(-1)
    return tokens


def query_nll_numpy(logits, targets) -> float:
    """Computes mean query-region next-token negative log-likelihood in NumPy."""
    arr = np.asarray(logits, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.int64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    picked = np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return float(-picked[..., QUERY0:].mean())


def report(framework: str, variant: str, n_weights: int, finite: bool,
           nll_before: float, nll_after: float) -> None:
    """Asserts weight parity and finiteness, then prints a summary."""
    assert n_weights == EXPECTED_WEIGHTS, (
        f'{framework}/{variant}: expected {EXPECTED_WEIGHTS} meta-weights, got {n_weights}'
    )
    assert finite and np.isfinite(nll_before) and np.isfinite(nll_after), (
        f'{framework}/{variant}: non-finite losses or query NLL '
        f'({nll_before}, {nll_after})'
    )
    print(
        f'{framework:8s} {variant:4s} weights={n_weights:4d} '
        f'heldout_nll={nll_before:.3f}->{nll_after:.3f}',
        flush=True,
    )
