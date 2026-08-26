"""Static construction replay.

This module owns the stage before concrete tree binding.  It deliberately
does not inspect a Node's current member tree when replaying statics: that
tree may contain ephemeral surgery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nodejax.core.definition import Construction, Def
from nodejax.frozendict import frozendict
from nodejax.core.node import Node, _is_node
from nodejax.struct import Struct


def _freeze_paths(values: Mapping) -> frozendict:
    """Freeze a nested path tree after its local assembly is complete."""
    return frozendict({
        name: (_freeze_paths(value)
               if issubclass(type(value), Mapping) else value)
        for name, value in values.items()
    })


def _nested(overrides: dict[str, Any]) -> frozendict:
    """Expand dotted overrides into an immutable descendant tree."""
    out: dict[str, Any] = {}
    for path, value in overrides.items():
        here = out
        parts = path.split('.')
        for part in parts[:-1]:
            here = here.setdefault(part, {})
        here[parts[-1]] = value
    return _freeze_paths(out)


def _merge(base: Mapping, changes: Mapping) -> frozendict:
    """Return the recursively merged immutable path tree."""
    out = dict(base)
    for name, value in changes.items():
        if (issubclass(type(value), Mapping)
                and issubclass(type(out.get(name)), Mapping)):
            out[name] = _merge(out[name], value)
        else:
            out[name] = value
    return frozendict(out)


def _definition(value: Any) -> Def:
    """Extract the raw definition returned by a construction factory."""
    if type(value) is Def:
        return value
    if _is_node(value):
        return value._def
    raise TypeError(
        f'a definition factory returned {type(value).__name__}, not a Node')


def _specialize_value(value: Any, named: Mapping,
                      wildcards: Mapping) -> Any:
    """Apply descendant overrides to one recorded construction value."""
    from nodejax.core.generic import Generic

    if _is_node(value):
        return Node(_specialize(value._def, named, wildcards))
    if type(value) is Generic:
        forwarded = dict(named)
        forwarded.update({f'*.{key}': item
                          for key, item in wildcards.items()})
        return value.specialize(**forwarded) if forwarded else value
    if issubclass(type(value), Mapping):
        unknown = set(named) - set(value)
        if unknown:
            raise TypeError(
                f'specialize: {sorted(unknown)} are not construction values')
        return type(value)({
            name: (_specialize_value(item, named.get(name, {}), wildcards)
                   if issubclass(type(named.get(name, {})), Mapping)
                   else named[name] if name in named
                   else _specialize_value(item, {}, wildcards))
            for name, item in value.items()
        })
    if named:
        raise TypeError('specialize: a scalar static has no descendants')
    return value


def specialize(definition: Def, overrides: dict[str, Any]) -> Def:
    """Replay a definition's canonical construction record."""
    construction = definition.construction
    if construction is None:
        raise TypeError(
            f"specialize: '{definition.name}' has no construction record")

    root = {name: value for name, value in overrides.items()
            if not name.startswith('*.')}
    new_wildcards = {name[2:]: value for name, value in overrides.items()
                     if name.startswith('*.')}
    requested = _nested(root)
    wildcards = {**construction.wildcards, **new_wildcards}
    arguments = dict(construction.arguments.__items__)
    descendants = {}

    for name, change in requested.items():
        if name not in arguments:
            descendants[name] = change
        elif issubclass(type(change), Mapping):
            arguments[name] = _specialize_value(
                arguments[name], change, wildcards)
        else:
            arguments[name] = change

    for name, value in list(arguments.items()):
        if name in requested:
            continue
        if name in wildcards:
            arguments[name] = wildcards[name]
        else:
            arguments[name] = _specialize_value(value, {}, wildcards)

    built = _definition(construction.factory(**arguments))
    named = _merge(construction.named, descendants)

    if named or wildcards:
        members = {}
        for name, member in built.members.__items__:
            changes = named.get(name, {})
            if changes or wildcards:
                members[name] = _specialize(member, changes, wildcards)
            else:
                members[name] = member
        unknown = set(named) - set(members)
        if unknown:
            raise TypeError(
                f"specialize: {sorted(unknown)} are not members of "
                f"'{built.name}'")
        if members:
            built = built.bind_members(Struct(**members))

    return built.copy(construction=Construction(
        factory=construction.factory,
        arguments=Struct(**arguments),
        named=named,
        wildcards=frozendict(wildcards),
    ))


def _specialize(definition: Def, named: Mapping,
                wildcards: Mapping) -> Def:
    """Recursive descendant replay without rebuilding override syntax."""
    construction = definition.construction
    if construction is None:
        if named:
            raise TypeError(
                f"specialize: '{definition.name}' has no static record")
        if not wildcards:
            return definition
        return definition
    overrides = dict(named)
    overrides.update({f'*.{key}': value for key, value in wildcards.items()})
    return specialize(definition, overrides)
