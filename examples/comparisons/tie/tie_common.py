"""Shared Markov token dataset, hyperparameters, and assertions for weight tying."""

import os

import numpy as np

SMOKE = os.environ.get('NODEJAX_SMOKE') == '1'
VOCAB, DIM = 8, 6
POSITIONS, STEPS = (32, 3) if SMOKE else (256, 200)
LR = 0.03
CONCENTRATION = 2.0


def make_data(rs: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Samples one Markov token stream and returns `(prev_tokens, next_tokens)` (`int32`)."""
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
    """Asserts optimizer table count, drift behavior, and loss finiteness, then prints."""
    assert np.isfinite(first) and np.isfinite(last), (
        f'{name}: non-finite training loss ({first} -> {last})'
    )
    if expect_tied:
        assert table_copies == 1, f'{name}: expected 1 table leaf, got {table_copies}'
        assert drift == 0.0, f'{name}: tied tables drifted by {drift:.2e}'
    else:
        assert table_copies == 2 and drift > 0.0, (
            f'{name}: expected duplicated tables to drift (copies={table_copies}, drift={drift:.2e})'
        )
    print(
        f'{name:14s} table_copies={table_copies} drift={drift:.2e} '
        f'loss {first:.2f} -> {last:.2f}',
        flush=True,
    )
