"""Human-readable text inspection of `Node` definitions, `Generic` builders, and bound PyTrees.

Provides tree renderers (`tree_view`, `summary`), static-argument extraction
(`statics_by_path`), and detailed structure-plus-leaf formatting (`describe`),
together with pipeline-friendly `print_tree` and `print_summary` helpers that
print and return their input unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import BuiltinFunctionType, FunctionType
from typing import Callable

import jax
import jax.numpy as jnp

from nodejax.core.generic import Generic, is_generic
from nodejax.core.node import BaseNode, _is_node
from nodejax.core.types import PyTree
from nodejax.frozendict import frozendict
from nodejax.struct import Struct


def _arguments(value: PyTree) -> Struct | None:
    if is_generic(value):
        return value.statics
    construction = value._def.construction
    return None if construction is None else construction.arguments


def statics_by_path(value: PyTree) -> frozendict:
    """Collects recorded static construction arguments across a definition tree keyed by dotted path."""
    out = {}

    def walk(current, prefix=''):
        arguments = _arguments(current)
        if arguments is None:
            return
        for name, item in arguments.__items__:
            path = f'{prefix}{name}'
            if _is_node(item) or is_generic(item):
                walk(item, path + '.')
            elif issubclass(type(item), Mapping):
                for child_name, child in item.items():
                    child_path = f'{path}.{child_name}'
                    if _is_node(child) or is_generic(child):
                        walk(child, child_path + '.')
                    else:
                        out[child_path] = child
            else:
                out[path] = item

    walk(value)
    return frozendict(out)


def _leaf(value: PyTree) -> str:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return f'{array.dtype}() = {array}'
    return f'{array.dtype}{tuple(array.shape)}'


def _data_lines(tree: PyTree, pad: str = '') -> tuple[str, ...]:
    if tree is None or (type(tree) is tuple and not tree):
        return (pad + '()',)
    rows = []
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        rows.append(f'{pad}{jax.tree_util.keystr(path).lstrip(".")} = {_leaf(leaf)}')
        total += int(jnp.asarray(leaf).size)
    rows.append(f'{pad}({len(rows)} leaves, {total} elements)')
    return tuple(rows)


def _static_words(node: BaseNode | Generic) -> str:
    from nodejax.core.binding import REQUIRED
    arguments = _arguments(node)
    if arguments is None:
        return ''
    words = []
    for name, value in arguments.__items__:
        if (_is_node(value) or is_generic(value)
                or issubclass(type(value), Mapping)):
            continue
        if value is REQUIRED:
            shown = '<unbound>'
        elif type(value) in (FunctionType, BuiltinFunctionType):
            shown = f'<{value.__name__}>'
        else:
            shown = repr(value)
            if len(shown) > 40:
                shown = f'<{type(value).__name__}>'
        words.append(f'{name}={shown}')
    return ', '.join(words)


def _definition_lines(node: BaseNode, key: str | None = None,
                      depth: int = 0) -> list[str]:
    roles = [name for name, active in (
        ('params', node.parametric), ('state', node.cyclic)) if active]
    label = f'{key}: ' if key else ''
    statics = _static_words(node)
    row = f'{"  " * depth}{label}{node.name} [{", ".join(roles) or "pure"}]'
    lines = [row + (f' ({statics})' if statics else '')]
    for name, member in node.members.__items__:
        lines.extend(_definition_lines(member, name, depth + 1))
    return lines


def _generic_lines(value: Generic, key: str | None = None,
                   depth: int = 0) -> list[str]:
    label = f'{key}: ' if key else ''
    lines = [f'{"  " * depth}{label}{value.name} [generic] ({_static_words(value)})']
    for name, child in value.statics.__items__:
        if is_generic(child):
            lines.extend(_generic_lines(child, name, depth + 1))
        elif _is_node(child):
            lines.extend(_definition_lines(child, name, depth + 1))
    return lines


def describe(value: PyTree) -> str:
    """Formats a definition hierarchy, static arguments, and bound leaf shapes and dtypes as text."""
    if _is_node(value):
        lines = _definition_lines(value)
        for role, tree in value._binding_items():
            lines.extend([f'{role}:', *_data_lines(tree, '  ')])
        return '\n'.join(lines)
    if is_generic(value):
        return '\n'.join(_generic_lines(value))
    return '\n'.join(_data_lines(value))


def _member_data(tree: PyTree, parent: BaseNode, name: str) -> PyTree:
    if parent._def.layout.transparent_member == name:
        return tree
    if issubclass(type(tree), Struct) and name in tree:
        return tree[name]
    return None


def tree_view(value: PyTree, *, max_depth: int | None = None,
              unicode: bool = True) -> str:
    """Renders the member hierarchy of a `BaseNode` as an ASCII or Unicode tree with `[PC]` role tags."""
    if is_generic(value):
        return describe(value)
    if not _is_node(value):
        return describe(value)
    binding = dict(value._binding_items())
    param = binding.get('param')
    state = binding.get('state')
    branch, last = ('├── ', '└── ') if unicode else ('|-- ', '`-- ')
    pipe, blank = ('│   ', '    ') if unicode else ('|   ', '    ')
    lines = [value.name]

    def walk(node, param_slice, state_slice, prefix, depth):
        if max_depth is not None and depth >= max_depth:
            if node.members:
                lines.append(f'{prefix}{last}... ({len(node.members)} members)')
            return
        items = list(node.members.__items__)
        for index, (name, member) in enumerate(items):
            final = index == len(items) - 1
            roles = 'P' * member.parametric + 'C' * member.cyclic
            lines.append(
                f'{prefix}{last if final else branch}{name}: {member.name}'
                + (f' [{roles}]' if roles else ''))
            walk(
                member, _member_data(param_slice, node, name),
                _member_data(state_slice, node, name),
                prefix + (blank if final else pipe), depth + 1)
    walk(value, param, state, '', 1)
    return '\n'.join(lines)


def summary(value: PyTree, *, max_depth: int | None = None,
            print_fn: Callable | None = None) -> str:
    """Combines `tree_view` with total bound parameter and state element counts."""
    text = tree_view(value, max_depth=max_depth)
    lines = [text]
    if _is_node(value):
        for role, tree in value._binding_items():
            count = sum(int(jnp.asarray(leaf).size)
                        for leaf in jax.tree.leaves(tree))
            label = 'parameters' if role == 'param' else role
            lines.append(f'{label}: {count}')
    result = '\n'.join(lines)
    if print_fn is not None:
        print_fn(result)
    return result


def print_tree(value: PyTree, **options: PyTree) -> PyTree:
    """Prints `tree_view(value, **options)` and returns `value` unchanged."""
    print(tree_view(value, **options))
    return value


def print_summary(value: PyTree, **options: PyTree) -> PyTree:
    """Prints `summary(value, **options)` and returns `value` unchanged."""
    print(summary(value, **options))
    return value
