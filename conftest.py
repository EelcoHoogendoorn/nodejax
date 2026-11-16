"""Requires `--full` before running broad test collections across the repository.

Targeted test runs stay fast during iteration, whereas broad collections above
`BROAD` tests require explicit opt-in via `--full` (for example `pytest -n 8 -q --full`).
"""

import pytest

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
        f'{len(items)} tests collected: this is the full suite, which takes '
        'minutes.\n'
        'While iterating, run the files you touched:\n'
        '    pytest nodejax/nn/tests/test_norm.py\n'
        'Before a commit, ask for it and run it in parallel:\n'
        '    pytest -n 8 -q --full')
