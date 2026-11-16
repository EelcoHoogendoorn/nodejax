"""Open `Generic` blueprints representing `@node` calls with unbound statics.

Calling a `@node` factory or `@transform` with omitted static parameters (or
with nested `Generic` arguments) produces a `Generic` value rather than raising
immediately. `Generic.specialize` overlays dotted static overrides and wildcard
defaults (`'*.dtype'`), returning another `Generic` while unbound static
arguments remain and building the concrete Node view as soon as the last static
argument is bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable

from nodejax.frozendict import frozendict
from nodejax.core.binding import REQUIRED, _unflatten_dot_paths
from nodejax.core.types import PyTree
from nodejax.struct import Struct

if TYPE_CHECKING:
    from nodejax.core.node import BaseNode


def is_generic(value: object) -> bool:
    """Reports whether `value` is an open `Generic` blueprint."""
    return type(value) is Generic


def _unbound(value: PyTree, prefix: str = '') -> tuple[str, ...]:
    if value is REQUIRED:
        return (prefix.rstrip('.'),)
    if is_generic(value):
        return tuple(f'{prefix}{path}' for path in value.unbound())
    if issubclass(type(value), Mapping):
        out = []
        for name, child in value.items():
            out.extend(_unbound(child, f'{prefix}{name}.'))
        return tuple(out)
    return ()


def _overlay(value: PyTree, changes: PyTree, wilds: dict[str, PyTree],
             path: str) -> PyTree:
    if is_generic(value):
        nested = changes if issubclass(type(changes), Mapping) else {}
        if changes is not None and not issubclass(type(changes), Mapping):
            raise TypeError(f'specialize: {path} names a Generic; supply its fields')
        forwarded = {**nested, **{f'*.{k}': v for k, v in wilds.items()}}
        return value.specialize(**forwarded) if forwarded else value
    from nodejax.core.node import _is_node
    if _is_node(value):
        nested = changes if issubclass(type(changes), Mapping) else {}
        if changes is not None and not issubclass(type(changes), Mapping):
            return changes
        forwarded = {**nested, **{f'*.{k}': v for k, v in wilds.items()}}
        return value.specialize(**forwarded) if forwarded else value
    if issubclass(type(value), Mapping):
        nested = {} if changes is None else changes
        if not issubclass(type(nested), Mapping):
            return changes
        unknown = set(nested) - set(value)
        if unknown:
            raise TypeError(
                f'specialize: {sorted(unknown)} not in {path or "mapping"!r}')
        return frozendict({
            name: _overlay(child, nested.get(name), wilds,
                           f'{path}.{name}' if path else name)
            for name, child in value.items()
        })
    if changes is not None:
        return changes
    return value


class Generic:
    """Represents a deferred `@node` or `@transform` call with unbound static arguments."""

    def __init__(self, name: str, factory: Callable, statics: Struct) -> None:
        self.name = name
        self.factory = factory
        self.statics = statics

    @property
    def generic(self) -> bool:
        return True

    def unbound(self) -> tuple[str, ...]:
        """Returns the dotted paths of all static arguments still awaiting values."""
        out = []
        for name, value in self.statics.__items__:
            out.extend(_unbound(value, f'{name}.'))
        return tuple(out)

    def specialize(self, **overrides: PyTree) -> BaseNode | Generic:
        """Applies static `overrides` and builds the Node once no unbound statics remain."""
        wilds = {key[2:]: value for key, value in overrides.items()
                 if key.startswith('*.')}
        named = _unflatten_dot_paths({key: value for key, value in overrides.items()
                                      if not key.startswith('*.')})
        own = {name for name, _ in self.statics.__items__}

        unknown = set(named) - own
        if unknown:
            raise TypeError(
                f"specialize: {sorted(unknown)} not in '{self.name}' record")

        take = {}
        for name, value in self.statics.__items__:
            change = named.get(name)
            if name in wilds and name not in named:
                change = wilds[name]
            updated = _overlay(value, change, wilds, name)
            if updated is not value:
                take[name] = updated

        if not take:
            return self
        built = self.factory(**{**dict(self.statics.__items__), **take})
        if is_generic(built):
            return built
        node = built.node if built.bound else built
        return node

    def statics_by_path(self) -> frozendict:
        """Returns a flattened `frozendict` of bound and unbound static paths."""
        from nodejax.core.printing import statics_by_path
        return statics_by_path(self)

    def describe(self) -> str:
        """Formats a multi-line structural description of this `Generic` blueprint."""
        from nodejax.core.printing import describe
        return describe(self)

    def tree_view(self) -> str:
        """Renders the static argument hierarchy of this `Generic` blueprint as an ASCII tree."""
        from nodejax.core.printing import tree_view
        return tree_view(self)

    def summary(self) -> str:
        """Formats a compact one-block summary of this `Generic` and its unbound statics."""
        from nodejax.core.printing import summary
        return summary(self)

    def __rshift__(self, other: BaseNode | Generic) -> BaseNode | Generic:
        from nodejax.core.compose import _compose
        return _compose(self, other)

    def __rrshift__(self, other: BaseNode | Generic) -> BaseNode | Generic:
        from nodejax.core.compose import _compose
        return _compose(other, self)

    def __getattr__(self, requested: str):
        if requested.startswith('__'):
            raise AttributeError(requested)
        raise TypeError(
            f"{self.name}: statics unbound {self.unbound()}, so there is no "
            f'{requested!r} yet; specialize supplies them and the factory '
            'builds the node')

    def __repr__(self) -> str:
        return f'Generic({self.name!r}, unbound={self.unbound()})'
