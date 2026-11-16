"""Import and reference-execution suite across `nodejax` and `examples`."""

import importlib
import pkgutil

import pytest

import examples
import nodejax

OPTIONAL = ('torch', 'tensorflow', 'keras', 'haiku', 'equinox', 'flax')


def _all_modules():
    for package in (nodejax, examples):
        for info in pkgutil.walk_packages(package.__path__, prefix=f'{package.__name__}.'):
            yield info.name


@pytest.mark.parametrize('name', sorted(_all_modules()))
def test_every_module_imports(name: str) -> None:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else ''
        if missing.split('.')[0] in OPTIONAL:
            pytest.skip(f'{name} requires optional dependency {missing}')
        raise


RUNNABLE = [
    'examples.comparisons.chunk.chunk_nodejax',
    'examples.comparisons.chunk.chunk_nnx',
    'examples.comparisons.chunk.chunk_linen',
    'examples.comparisons.chunk.chunk_equinox',
    'examples.comparisons.chunk.chunk_haiku',
    'examples.comparisons.chunk.chunk_torch',
    'examples.comparisons.generics.generics_nodejax',
    'examples.comparisons.generics.generics_nnx',
    'examples.comparisons.generics.generics_equinox',
    'examples.comparisons.generics.generics_keras',
    'examples.comparisons.generics.generics_torch',
    'examples.comparisons.imu.imu_nodejax',
    'examples.comparisons.imu.imu_nnx',
    'examples.comparisons.imu.imu_equinox',
    'examples.comparisons.lift.lift_nodejax',
    'examples.comparisons.lift.lift_nnx',
    'examples.comparisons.lift.lift_equinox',
    'examples.comparisons.mode.mode_nodejax',
    'examples.comparisons.mode.mode_nnx',
    'examples.comparisons.mode.mode_equinox',
    'examples.comparisons.mode.mode_haiku',
    'examples.comparisons.mode.mode_torch',
    'examples.comparisons.residual.residual_nodejax',
    'examples.comparisons.residual.residual_nnx',
    'examples.comparisons.residual.residual_equinox',
    'examples.comparisons.tie.tie_nodejax',
    'examples.comparisons.tie.tie_nnx',
    'examples.comparisons.tie.tie_equinox',
    'examples.comparisons.tie.tie_haiku',
    'examples.comparisons.tie.tie_torch',
    'examples.comparisons.tower.tower_nodejax',
    'examples.comparisons.tower.tower_nnx',
    'examples.comparisons.tower.tower_nnx_reusable',
    'examples.comparisons.tower.tower_equinox',
    'examples.comparisons.tower.tower_keras',
    'examples.comparisons.tower.tower_torch',
    'examples.comparisons.ttt.ttt_nodejax',
    'examples.comparisons.ttt.ttt_jax',
    'examples.comparisons.ttt.ttt_nnx',
    'examples.comparisons.ttt.ttt_equinox',
    'examples.comparisons.ttt.ttt_haiku',
    'examples.comparisons.ttt.ttt_torch',
]


@pytest.mark.parametrize('name', RUNNABLE)
def test_comparison_reproduces_reference(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else ''
        if missing.split('.')[0] in OPTIONAL:
            pytest.skip(f'{name} requires optional dependency {missing}')
        raise
    module.main()
