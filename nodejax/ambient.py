"""Ambient construction arguments: dynamic scope for the def-building
stage, declared at the definition site.

The problem: quantities like dt or a train flag are needed by a dozen
def factories across a construction graph, and threading them through
every call site is noise (while registries of wrapped factories are
worse). The solution is dynamic scope, tightly fenced:

    @ambient                       # at the definition site
    def pid_def(dt, dwrap=None): ...

    with ambient(dt=1e-4):         # at the single point of use
        pid_def()                  # dt filled from scope

Rules that keep it fenced:
- Eligibility is declared by the decorator, visible where the factory is
  defined; undecorated functions never see the scope.
- Explicit arguments ALWAYS win; the scope only fills parameters the
  call left unbound.
- Outside any scope, an unfilled required parameter fails exactly as it
  always did (a TypeError at the factory) — nothing becomes optional.
- CONSTRUCTION TIME ONLY: factories run when defs are built; nothing
  here exists at trace/apply time, so the functional semantics of nodes
  are untouched. This is scoping for Python configuration, not implicit
  state in the compute graph.
- Scopes nest (inner wins) and are contextvar-based (task/thread safe).

Relationship to '*.name' generic broadcasts: broadcasts travel WITH a
generic tree and resolve at specialize, wherever and whenever that
happens; ambient scope covers plain-callable construction happening
lexically inside the with-block. Deferred specialization outside the
block should use broadcasts.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

_SCOPE: ContextVar[tuple[dict[str, Any], ...]] = ContextVar('ambient_scope', default=())

_FILLABLE = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@contextmanager
def _scope(values: dict[str, Any]):
    token = _SCOPE.set(_SCOPE.get() + (values,))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _lookup(name: str) -> tuple[bool, Any]:
    for frame in reversed(_SCOPE.get()):
        if name in frame:
            return True, frame[name]
    return False, None


class _Ambient:
    """`ambient` is both the decorator (given a function) and the scope
    (given keyword values)."""

    def __call__(self, fn: Callable | None = None, **values: Any):
        if fn is not None:
            if values:
                raise TypeError('ambient takes a function OR scope values, not both')
            sig = inspect.signature(fn)
            names = [n for n, p in sig.parameters.items() if p.kind in _FILLABLE]

            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any):
                taken = sig.bind_partial(*args, **kwargs).arguments
                filled = dict(kwargs)
                for name in names:
                    if name not in taken:
                        hit, value = _lookup(name)
                        if hit:
                            filled[name] = value
                return fn(*args, **filled)

            return wrapper
        return _scope(values)


ambient = _Ambient()
