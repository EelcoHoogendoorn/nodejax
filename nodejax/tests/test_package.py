"""Verifies package version alignment and PEP 561 `py.typed` marker presence."""

import pathlib
import tomllib

import nodejax


def test_version_matches_pyproject() -> None:
    """Checks that `nodejax.__version__` matches `[project].version` in `pyproject.toml`."""
    repo_root = pathlib.Path(nodejax.__file__).resolve().parent.parent
    pyproject = repo_root / 'pyproject.toml'
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
        assert nodejax.__version__ == data['project']['version']
    assert isinstance(nodejax.__version__, str) and nodejax.__version__


def test_py_typed_marker_exists() -> None:
    """Checks that the PEP 561 `py.typed` marker ships inside `nodejax/`."""
    marker = pathlib.Path(nodejax.__file__).resolve().parent / 'py.typed'
    assert marker.is_file()
