"""Configures pytest collection guards, smoke-test budgets, and single-threaded worker pools.

Sets `NODEJAX_SMOKE=1` so example and comparison tests execute short coverage runs rather than full training demos, and caps OpenMP/MKL threads per worker for parallel `pytest -n` runs. Broad collections above `BROAD` tests require `--full`.
"""

import os

import pytest

os.environ.setdefault('NODEJAX_SMOKE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Threshold above which a pytest invocation is treated as a full-suite run.
BROAD = 300


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--full', action='store_true', default=False,
        help='run the whole suite; expected before a commit, not while '
             'iterating')


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption('--full') or len(items) <= BROAD:
        return
    raise pytest.UsageError(
        f'{len(items)} tests collected: this is the full suite.\n'
        'While iterating, run the files you touched:\n'
        '    pytest nodejax/nn/tests/test_norm.py\n'
        'Before a commit, ask for it and run it in parallel:\n'
        '    pytest -n 8 -q --full')
