"""Construction-time scope and definition-factory recording."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import Any, Callable

from nodejax.core.definition import Construction
from nodejax.frozendict import frozendict
from nodejax.core.generic import Generic, is_generic
from nodejax.core.node import Node, _is_node
from nodejax.struct import Struct


_SCOPE: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    'nodejax_ambient', default=())
_FILLABLE = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


@contextmanager
def _scope(values):
    token = _SCOPE.set(_SCOPE.get() + (values,))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _lookup(name):
    for scope in reversed(_SCOPE.get()):
        if name in scope:
            return True, scope[name]
    return False, None


class _Ambient:
    def __call__(self, fn=None, **values):
        if fn is not None:
            raise TypeError('use @node on factories and ambient(...) as scope')
        return _scope(values)


ambient = _Ambient()


def _snake(name):
    return re.sub(
        r'([a-z0-9])([A-Z])', r'\1_\2',
        re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name),
    ).lower()


def _contains_unbound(value):
    from nodejax.core.binding import REQUIRED

    if value is REQUIRED or is_generic(value):
        return True
    if type(value) is Struct:
        return any(_contains_unbound(item) for item in value)
    if issubclass(type(value), Mapping):
        return any(_contains_unbound(item) for item in value.values())
    if type(value) in (tuple, list):
        return any(_contains_unbound(item) for item in value)
    return False


def _construction_value(value):
    if _is_node(value):
        return value.node
    if issubclass(type(value), Mapping):
        return frozendict({
            name: _construction_value(item) for name, item in value.items()
        })
    if type(value) is tuple:
        return tuple(_construction_value(item) for item in value)
    if type(value) is list:
        return tuple(_construction_value(item) for item in value)
    return value


def node(fn: Callable | None = None, *, name: str | None = None):
    """Record a factory call, deferring execution while statics are missing."""
    if fn is None:
        return partial(node, name=name)
    signature = inspect.signature(fn)
    fillable = [field for field, parameter in signature.parameters.items()
                if parameter.kind in _FILLABLE]

    @wraps(fn)
    def factory(*args, **kwargs):
        supplied = dict(kwargs)
        try:
            partial_call = signature.bind_partial(*args, **kwargs)
        except TypeError:
            partial_call = None
        if partial_call is not None:
            for field in fillable:
                if field not in partial_call.arguments:
                    found, value = _lookup(field)
                    if found:
                        supplied[field] = value
        try:
            call = signature.bind_partial(*args, **supplied)
        except TypeError:
            call = None

        if call is not None:
            from nodejax.core.binding import REQUIRED
            missing = [
                field for field, parameter in signature.parameters.items()
                if parameter.kind in _FILLABLE
                and ((parameter.default is inspect.Parameter.empty
                      and field not in call.arguments)
                     or call.arguments.get(field) is REQUIRED)
            ]
            if missing or any(_contains_unbound(value)
                              for value in call.arguments.values()):
                call.apply_defaults()
                arguments = {
                    **{field: _construction_value(value)
                       for field, value in call.arguments.items()},
                    **{field: REQUIRED for field in missing},
                }
                return Generic(_snake(fn.__name__), factory,
                               Struct(**arguments))
            product = fn(*call.args, **call.kwargs)
            call.apply_defaults()
            recorded = Struct(**{
                field: _construction_value(value)
                for field, value in call.arguments.items()
            })
        else:
            product = fn(*args, **supplied)
            if args:
                return product
            recorded = Struct(**{
                field: _construction_value(value)
                for field, value in supplied.items()
            })

        if is_generic(product):
            return Generic(name or product.name, factory, recorded)

        if not _is_node(product):
            return product

        definition = product._def
        construction = definition.construction
        if construction is not None and construction.factory is factory:
            # An inner call of this same factory built the product and
            # recorded the arguments it was actually built with.
            recorded = construction.arguments
        unnamed_wrapper = (
            definition.layout.transparent_member is not None
            and construction is not None
            and 'name' in construction.arguments
            and construction.arguments.name is None
        )
        weak_name = (definition.name in ('apply', '<lambda>', 'composite')
                     or definition.name.startswith('composite(')
                     or unnamed_wrapper)
        actual_name = (name if name is not None else
                       _snake(fn.__name__) if weak_name else definition.name)

        member_arguments = {
            member: field
            for member in definition.members.__keys__
            for field, value in recorded.__items__
            if field == member
        }
        tree = definition.tree
        if len(member_arguments) == len(definition.members):
            def tree(replacements):
                arguments = dict(recorded.__items__)
                for member, field in member_arguments.items():
                    arguments[field] = Node(replacements[member])
                rebuilt = factory(**arguments)
                if not _is_node(rebuilt):
                    raise TypeError('tree binding did not rebuild a Node')
                return rebuilt._def

        definition = definition.copy(
            name=actual_name,
            construction=Construction(factory, recorded),
            tree=tree,
        )
        return product._with_definition(definition)

    return factory
