"""Make the full suite something you ask for on purpose.

The rule is targeted tests while iterating, the full suite only before a
commit. A four minute run after a one line change is the expensive mistake,
not the extra targeted run, so a broad collection without ``--full`` stops
here rather than costing that time.

Breadth is measured by how many tests were collected, not by how the target
was spelled, so broad collections across ``nodejax`` and ``examples`` are
caught the same way.
"""

import pytest

#: Above this many collected tests, a run is a whole-suite run in all but
#: name. The largest single file sits well under it, so any one file, class
#: or directory of a few files still runs without ceremony.
BROAD = 300


def pytest_addoption(parser):
    parser.addoption(
        '--full', action='store_true', default=False,
        help='run the whole suite; expected before a commit, not while '
             'iterating')


def pytest_collection_modifyitems(session, config, items):
    if config.getoption('--full') or len(items) <= BROAD:
        return
    raise pytest.UsageError(
        f'{len(items)} tests collected: this is the full suite, which takes '
        'minutes.\n'
        'While iterating, run the files you touched:\n'
        '    pytest nodejax/nn/tests/test_norm.py\n'
        'Before a commit, ask for it and run it in parallel:\n'
        '    pytest -n 8 -q --full')
