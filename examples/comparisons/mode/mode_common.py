"""Shared dataset, hyperparameters, and assertions for the train/eval mode comparison."""

import numpy as np

DIM, HIDDEN, CLASSES = 8, 16, 3
N, STEPS = 128, 200
RATE, MOMENTUM, LR = 0.3, 0.1, 0.02


def make_data(rs: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Samples `N` points from `CLASSES` Gaussian clusters in `DIM` dimensions."""
    centers = 2.0 * rs.standard_normal((CLASSES, DIM))
    labels = rs.randint(CLASSES, size=N).astype(np.int32)
    xs = (centers[labels] + rs.standard_normal((N, DIM))).astype(np.float32)
    return xs, labels


def report(name: str, train_stochastic: bool, eval_deterministic: bool,
           eval_isolated: bool, first: float, last: float) -> None:
    """Asserts all three mode invariants and training progress, then prints a summary."""
    assert train_stochastic, f'{name}: training forward pass was deterministic'
    assert eval_deterministic, f'{name}: evaluation forward pass was stochastic'
    assert eval_isolated, f'{name}: evaluation logits leaked batch-peer statistics'
    assert np.isfinite(first) and np.isfinite(last) and last < first, (
        f'{name}: training loss failed to decrease ({first:.2f} -> {last:.2f})'
    )
    print(
        f'{name:10s} train_stochastic={train_stochastic} '
        f'eval_deterministic={eval_deterministic} eval_isolated={eval_isolated} '
        f'loss {first:.2f} -> {last:.2f}',
        flush=True,
    )
