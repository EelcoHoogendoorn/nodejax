"""Deferred factory calls with unbound static arguments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from nodejax.frozendict import frozendict
from nodejax.binding import REQUIRED, _unflatten_dot_paths
from nodejax.struct import Struct


def is_generic(value: Any) -> bool:
    """Return whether ``value`` is a deferred factory call."""
    return type(value) is Generic


def _unbound(value: Any, prefix: str = '') -> tuple[str, ...]:
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


def _overlay(value: Any, changes: Any, wilds: dict[str, Any], path: str) -> Any:
    """Overlay one recursive construction value without dropping siblings."""
    if is_generic(value):
        nested = changes if issubclass(type(changes), Mapping) else {}
        if changes is not None and not issubclass(type(changes), Mapping):
            raise TypeError(f'specialize: {path} names a Generic; supply its fields')
        forwarded = {**nested, **{f'*.{k}': v for k, v in wilds.items()}}
        return value.specialize(**forwarded) if forwarded else value
    from nodejax.node import _is_node
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
    """A factory call whose static arguments are not all bound yet."""

    def __init__(self, name: str, factory: Callable, statics: Struct):
        self.name = name
        self.factory = factory
        self.statics = statics

    @property
    def generic(self) -> bool:
        """Whether this value represents an unfinished construction."""
        return True

    def unbound(self) -> tuple[str, ...]:
        """Return dot-separated paths of missing static arguments."""
        out = []
        for name, value in self.statics.__items__:
            out.extend(_unbound(value, f'{name}.'))
        return tuple(out)

    def specialize(self, **overrides: Any) -> Any:
        """Apply static overrides and rerun the factory when complete."""
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
        # Specialization returns an unbound definition view.
        node = built.node if built.bound else built
        return node

    def statics_by_path(self) -> frozendict:
        """Return the flattened static argument graph."""
        from nodejax.printing import statics_by_path
        return statics_by_path(self)

    def describe(self) -> str:
        """Describe the deferred construction."""
        from nodejax.printing import describe
        return describe(self)

    def tree_view(self) -> str:
        """Render the static argument graph as a tree."""
        from nodejax.printing import tree_view
        return tree_view(self)

    def summary(self) -> str:
        """Summarize the construction and its missing arguments."""
        from nodejax.printing import summary
        return summary(self)

    def __rshift__(self, other: Any) -> Any:
        """Compose while deferring the resulting construction."""
        from nodejax.compose import _compose
        return _compose(self, other)

    def __rrshift__(self, other: Any) -> Any:
        from nodejax.compose import _compose
        return _compose(other, self)

    def __getattr__(self, requested: str) -> Any:
        """Reject Node operations until all statics are bound."""
        if requested.startswith('__'):
            raise AttributeError(requested)
        raise TypeError(
            f"{self.name}: statics unbound {self.unbound()}, so there is no "
            f'{requested!r} yet; specialize supplies them and the factory '
            'builds the node')

    def __repr__(self) -> str:
        return f'Generic({self.name!r}, unbound={self.unbound()})'
