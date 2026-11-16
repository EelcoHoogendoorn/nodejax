"""Shared Markov token dataset, hyperparameters, and assertions for weight tying."""

import numpy as np

VOCAB, DIM = 8, 6
POSITIONS, STEPS = 256, 200
LR = 0.03
CONCENTRATION = 2.0


def make_data(rs: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Sample one Markov token stream and return `(prev_tokens, next_tokens)` (`int32`)."""
    logits = CONCENTRATION * rs.standard_normal((VOCAB, VOCAB))
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    tokens = np.zeros(POSITIONS + 1, dtype=np.int32)
    state = int(rs.randint(VOCAB))
    for t in range(POSITIONS + 1):
        tokens[t] = state
        state = int((probs[state].cumsum() > rs.random()).argmax())
    return tokens[:-1], tokens[1:]


def report(name: str, table_copies: int, drift: float, first: float,
           last: float, *, expect_tied: bool = True) -> None:
    """Assert optimizer table count, drift behavior, and loss reduction, then print."""
    assert np.isfinite(first) and np.isfinite(last) and last < first, (
        f'{name}: training loss failed to decrease ({first:.2f} -> {last:.2f})'
    )
    if expect_tied:
        assert table_copies == 1, f'{name}: expected 1 table leaf, got {table_copies}'
        assert drift == 0.0, f'{name}: tied tables drifted by {drift:.2e}'
    else:
        assert table_copies == 2 and drift > 1e-3, (
            f'{name}: expected duplicated tables to drift (copies={table_copies}, drift={drift:.2e})'
        )
    print(
        f'{name:14s} table_copies={table_copies} drift={drift:.2e} '
        f'loss {first:.2f} -> {last:.2f}',
        flush=True,
    )
