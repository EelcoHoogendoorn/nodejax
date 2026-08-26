"""Every module imports, and the runnable examples run.

The suite collects `test_*.py` and nothing else, so a third of the example
tree was never loaded: the comparison modules are named for what they compare
rather than for pytest, and the actuator package is example code beneath its
own tests. Renaming a transform could therefore leave them calling a name that
no longer exists, with the suite green.

That is not hypothetical. `frozen` was renamed and chunk_nodejax kept
importing the old name; `PNode.scan` was retired and tower_nodejax went on
calling it, unnoticed for as long as it took someone to run the file by hand.
Both are the same gap, and this file is the patch.

Two tiers, because they cost differently. Importing every module is cheap and
catches the whole class above. Running the comparisons costs seconds each, and
buys the assertions they carry: each one reproduces a reference computed by
hand in its `_common` module, so a wrong answer fails rather than printing.
"""

import importlib
import pkgutil

import pytest

import examples
import nodejax

# Modules whose imports are not ours to satisfy. The cross-framework
# comparisons exist to be read beside the nodejax file as much as run, and a
# missing torch should not fail this suite.
OPTIONAL = ('torch', 'tensorflow', 'keras', 'haiku', 'equinox', 'flax')


def _all_modules():
    """Every module in both packages, tests included: importing a test module is
    as good a check as importing anything else."""
    for package in (nodejax, examples):
        for info in pkgutil.walk_packages(package.__path__, prefix=f'{package.__name__}.'):
            yield info.name


@pytest.mark.parametrize('name', sorted(_all_modules()))
def test_every_module_imports(name):
    """The cheap tier, and the one that catches a rename sweep that missed a
    file. A module that needs an absent optional framework skips; anything
    else that fails to import is a break."""
    try:
        importlib.import_module(name)
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else ''
        if missing.split('.')[0] in OPTIONAL:
            pytest.skip(f'{name} needs {missing}, which is not installed')
        raise


# The nodejax side of each comparison, which needs no optional framework. Each
# main() asserts against a reference its _common module computes independently,
# so running it is a real check and not a smoke test.
RUNNABLE = [
    # A direct transform-authoring comparison rather than a model benchmark:
    # both columns build one reusable stack primitive and exercise independent
    # construction, mutable per-layer state, apply randomness, and aux.
    'examples.comparisons.lift.lift_nodejax',
    'examples.comparisons.lift.lift_nnx',
    'examples.comparisons.lift.lift_equinox',
    # The transparent-wrapper family: the smallest possible transform, so a
    # column that cannot forward its member's facts has no axis machinery to
    # blame it on.
    'examples.comparisons.residual.residual_nodejax',
    'examples.comparisons.residual.residual_nnx',
    'examples.comparisons.residual.residual_equinox',
    'examples.comparisons.chunk.chunk_nodejax',
    'examples.comparisons.tower.tower_nodejax',
    'examples.comparisons.ttt.ttt_nodejax',
    # the other four columns of the chunk comparison. They assert against the
    # same references the nodejax one does, and running them here is the only
    # thing that checks it: report() prints a verdict, and a printed verdict
    # in a file nobody runs is not a check. flax was silently wrong for a
    # commit because of exactly that.
    'examples.comparisons.chunk.chunk_flax',
    'examples.comparisons.chunk.chunk_flax_linen',
    'examples.comparisons.chunk.chunk_equinox',
    'examples.comparisons.chunk.chunk_haiku',
    'examples.comparisons.chunk.chunk_torch',
    # the ttt rivals. Self-contained runnable files rot silently unless run:
    # two of these five were broken for days by call-site sweeps, their
    # mains guarded out of every other test. They assert little internally
    # (torch checks its meta loss fell), so this tier is mostly the
    # does-it-still-run check, which is exactly the check they lacked.
    'examples.comparisons.ttt.ttt_rnn_by_hand',
    'examples.comparisons.ttt.ttt_equinox',
    'examples.comparisons.ttt.ttt_haiku',
    'examples.comparisons.ttt.ttt_rnn_flax',
    'examples.comparisons.ttt.ttt_rnn_torch',
    # the tower rivals, same reasoning: each main() asserts its meta loss
    # fell, and nothing else ever runs them
    'examples.comparisons.tower.tower_flax',
    'examples.comparisons.tower.tower_flax_reusable',
    'examples.comparisons.tower.tower_equinox',
    'examples.comparisons.tower.tower_torch',
    'examples.comparisons.tower.tower_keras',
    # the tie family: sharing as a property you can lose, each column
    # printing how many table copies its optimizer saw and the drift
    'examples.comparisons.tie.tie_nodejax',
    'examples.comparisons.tie.tie_equinox',
    'examples.comparisons.tie.tie_flax',
    'examples.comparisons.tie.tie_torch',
    'examples.comparisons.tie.tie_haiku',
    # the mode family: train and eval as a property of the program
    'examples.comparisons.mode.mode_nodejax',
    'examples.comparisons.mode.mode_flax',
    'examples.comparisons.mode.mode_equinox',
    'examples.comparisons.mode.mode_haiku',
    'examples.comparisons.mode.mode_torch',
    # the generics family: configuring a deep composition, where every
    # column's report() asserts the SAME parameter count from the shared
    # formula, so a column that built a different architecture fails
    # here instead of printing a prettier number nobody checked
    'examples.comparisons.generics.generics_nodejax',
    'examples.comparisons.generics.generics_equinox',
    'examples.comparisons.generics.generics_flax',
    'examples.comparisons.generics.generics_keras',
    'examples.comparisons.generics.generics_torch',
]
# imu_nodejax is not here because it needs no help: pytest.ini already names it
# in python_files, so its own test functions are collected. That line is the
# existing patch for one file of this gap, and the reason it exists is the
# reason the rest of this module does.


@pytest.mark.parametrize('name', RUNNABLE)
def test_the_comparison_reproduces_its_reference(name):
    """The second tier. These print a table when run by hand, and the printing
    is not the point: the assertions inside main() are, and they compare
    against plain-jax references written without the framework."""
    try:
        module = importlib.import_module(name)
    except ImportError as e:                     # an absent optional framework
        missing = str(e).split("'")[1] if "'" in str(e) else ''
        if missing.split('.')[0] in OPTIONAL:
            pytest.skip(f'{name} needs {missing}, which is not installed')
        raise
    module.main()
