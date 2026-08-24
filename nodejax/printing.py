"""Text views of definitions and their bound data."""

from __future__ import annotations

from collections.abc import Mapping
from types import BuiltinFunctionType, FunctionType
from typing import Any

import jax
import jax.numpy as jnp

from nodejax.frozendict import frozendict
from nodejax.generic import Generic, is_generic
from nodejax.node import BaseNode, _is_node
from nodejax.struct import Struct


def _arguments(value):
    if is_generic(value):
        return value.statics
    construction = value._def.construction
    return None if construction is None else construction.arguments


def statics_by_path(value) -> frozendict:
    """Flatten the recursive canonical construction record by dotted path."""
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


def _leaf(value):
    array = jnp.asarray(value)
    if array.ndim == 0:
        return f'{array.dtype}() = {array}'
    return f'{array.dtype}{tuple(array.shape)}'


def _data_lines(tree, pad=''):
    if tree is None or (type(tree) is tuple and not tree):
        return (pad + '()',)
    rows = []
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        rows.append(f'{pad}{jax.tree_util.keystr(path).lstrip(".")} = {_leaf(leaf)}')
        total += int(jnp.asarray(leaf).size)
    rows.append(f'{pad}({len(rows)} leaves, {total} elements)')
    return tuple(rows)


def _static_words(node):
    from nodejax.binding import REQUIRED
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


def _definition_lines(node: BaseNode, key=None, depth=0):
    roles = [name for name, active in (
        ('params', node.parametric), ('state', node.cyclic)) if active]
    label = f'{key}: ' if key else ''
    statics = _static_words(node)
    row = f'{"  " * depth}{label}{node.name} [{", ".join(roles) or "pure"}]'
    lines = [row + (f' ({statics})' if statics else '')]
    for name, member in node.members.__items__:
        lines.extend(_definition_lines(member, name, depth + 1))
    return lines


def _generic_lines(value: Generic, key=None, depth=0):
    label = f'{key}: ' if key else ''
    lines = [f'{"  " * depth}{label}{value.name} [generic] ({_static_words(value)})']
    for name, child in value.statics.__items__:
        if is_generic(child):
            lines.extend(_generic_lines(child, name, depth + 1))
        elif _is_node(child):
            lines.extend(_definition_lines(child, name, depth + 1))
    return lines


def describe(value) -> str:
    if _is_node(value):
        lines = _definition_lines(value)
        for role, tree in value._binding_items():
            lines.extend([f'{role}:', *_data_lines(tree, '  ')])
        return '\n'.join(lines)
    if is_generic(value):
        return '\n'.join(_generic_lines(value))
    return '\n'.join(_data_lines(value))


def _member_data(tree, parent, name):
    if parent._def.layout.transparent_member == name:
        return tree
    if issubclass(type(tree), Struct) and name in tree:
        return tree[name]
    return None


def tree_view(value, *, max_depth=None, unicode=True) -> str:
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

    def walk(node, p, s, prefix, depth):
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
                member, _member_data(p, node, name),
                _member_data(s, node, name),
                prefix + (blank if final else pipe), depth + 1)
    walk(value, param, state, '', 1)
    return '\n'.join(lines)


def summary(value, *, max_depth=None, print_fn=None) -> str:
    """Compact definition tree plus aggregate bound-data sizes."""
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


def print_tree(value, **options):
    print(tree_view(value, **options))
    return value


def print_summary(value, **options):
    print(summary(value, **options))
    return value
