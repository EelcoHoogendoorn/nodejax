"""Lower ordinary authored functions into canonical definition calls."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.core.author_view import AuthorNode
from nodejax.core.binding import (
    _METHOD_CHANNELS, _bundle_spec, _has_rng_field,
)
from nodejax.core.contract import ApplyCall, CallForm, InitCall, ParamCall
from nodejax.frozendict import frozendict
from nodejax.core.rng import KeyStream, MaybeKeyStream
from nodejax.struct import Struct


def _keys_only(tree: Any, where: str) -> Any:
    """Store raw keys under ``rng`` and reject other capability escapes.

    Authored output containers keep their declared pytree type. A returned
    ``dict`` or ``list`` is therefore user output, not framework working
    storage.
    """
    def walk(value: Any, path: str, rng_field: bool = False):
        if type(value) is KeyStream:
            if rng_field:
                return value.next()
            raise TypeError(
                f'{where}: a KeyStream escaped at {path or "the output"}; '
                'draw with rng.next()')
        if type(value) is MaybeKeyStream:
            raise TypeError(
                f'{where}: a MaybeKeyStream escaped at '
                f'{path or "the output"}')
        if issubclass(type(value), Struct):
            return type(value)(**{
                name: walk(child, f'{path}.{name}' if path else name,
                           name == 'rng')
                for name, child in value.__items__
            })
        if type(value) is dict:
            return {
                name: walk(child, f'{path}.{name}' if path else str(name),
                           name == 'rng')
                for name, child in value.items()
            }
        if issubclass(type(value), tuple):
            children = tuple(walk(child, f'{path}[{i}]')
                             for i, child in enumerate(value))
            return (children if type(value) is tuple
                    else type(value)(*children))
        if type(value) is list:
            return [walk(child, f'{path}[{i}]')
                    for i, child in enumerate(value)]
        return value

    return walk(tree, '')


def tree_asarray(tree: Any) -> Any:
    """Convert every leaf in a pytree with ``jnp.asarray``."""
    return jax.tree.map(jnp.asarray, tree)


def _signature(fn: Callable, role: str) -> Mapping:
    params = inspect.signature(fn).parameters
    if any(parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
    ) for parameter in params.values()):
        raise TypeError(f'{role} must declare a closed set of named fields')
    return params


_APPLY_CHANNELS = frozenset({'param', 'state', 'node'})


def _compile_apply(fn: Callable, *, parametric: bool, cyclic: bool,
                   owner: str) -> ApplyCall:
    signature = _signature(fn, 'apply')
    if 'self' in signature:
        raise TypeError(
            f'{owner}: leaf apply does not accept self; use explicit param '
            'and state')
    if 'param' in signature and not parametric:
        raise TypeError(
            f'{owner}: apply names param but no param constructor exists')
    if 'state' in signature and not cyclic:
        raise TypeError(
            f'{owner}: apply names state but no initializer exists')
    fields = [name for name in signature
              if name not in _APPLY_CHANNELS and name != 'rng']

    declaration = _bundle_spec(
        signature, drop=tuple(_APPLY_CHANNELS), owner=fn.__name__,
        allow_defaults=False)
    takes_rng = 'rng' in declaration
    if takes_rng:
        declaration = declaration.without('rng')
    has_state = 'state' in signature
    reads_def = 'node' in signature

    def impl(definition, param, state, formed_input, rng):
        arguments = {}
        for name in signature:
            if name == 'node':
                arguments[name] = AuthorNode(definition)
            elif name == 'param':
                arguments[name] = param
            elif name == 'state':
                arguments[name] = state
            elif name == 'rng':
                arguments[name] = rng._require()
            elif name in formed_input:
                arguments[name] = formed_input[name]
        output = _keys_only(fn(**arguments), f'{definition.name}.apply')
        return output if has_state else (state, output)

    if has_state:
        impl = _advance_state_rng(impl)
    return ApplyCall(
        impl=impl,
        form=CallForm.from_values(declaration),
        input_spec=Struct() if not fields else None,
        takes_rng=takes_rng,
        reads_def=reads_def,
    )


def _advance_state_rng(apply_impl: Callable) -> Callable:
    def wrapped(definition, param, state, formed_input, rng):
        if _has_rng_field(state):
            next_key, use_key = jax.random.split(state.rng)
            state, output = apply_impl(
                definition, param, state.replace(rng=use_key),
                formed_input, rng)
            return state.replace(rng=next_key), output
        return apply_impl(definition, param, state, formed_input, rng)
    return wrapped


def _compile_param(fn: Callable) -> ParamCall:
    signature = _signature(fn, 'parameter constructor')
    if 'input' in signature:
        raise TypeError('parameter constructors read shape through node.input')
    declaration = _bundle_spec(signature, drop=('node', 'param'))
    takes_rng = 'rng' in declaration
    if takes_rng:
        declaration = declaration.without('rng')

    def impl(definition, formed_input, rng):
        arguments = {}
        for name in signature:
            if name == 'node':
                arguments[name] = AuthorNode(definition)
            elif name == 'rng':
                arguments[name] = rng._require()
            elif name == 'param':
                arguments[name] = formed_input
            elif name in formed_input:
                arguments[name] = formed_input[name]
        return tree_asarray(_keys_only(
            fn(**arguments), f'{definition.name}.param'))

    return ParamCall(
        impl=impl,
        form=CallForm.from_values(declaration),
        takes_rng=takes_rng,
        reads_def='node' in signature,
    )


def _compile_init(fn: Callable, *, owner: str | None = None) -> InitCall:
    signature = _signature(fn, 'initializer')
    if 'self' in signature:
        raise TypeError(
            f'{owner}: authored init takes (param, input); '
            'self is the wired apply view')
    if ('input' in signature and
            signature['input'].default is not inspect.Parameter.empty):
        raise TypeError('init input is a required priming value or is omitted')
    primes = 'input' in signature
    object_names = ('param',)
    declaration = _bundle_spec(
        signature, drop=object_names + ('node', 'input', 'state'))
    takes_rng = 'rng' in declaration
    if takes_rng:
        declaration = declaration.without('rng')

    def filled(definition, param):
        """The init param names every declared member: a member
        without parameters answers the empty slot, so authored inits
        need not fork on a member's parametricity."""
        members = definition.members
        if not tuple(members.__keys__):
            return param
        held = tuple(param.__keys__) if type(param) is Struct else ()
        return Struct(**{
            name: (param[name] if name in held else ())
            for name in members.__keys__
        })

    def arguments(definition, param, formed_input, rng):
        out = {}
        for name in signature:
            if name in object_names:
                out[name] = filled(definition, param)
            elif name == 'node':
                out[name] = AuthorNode(definition)
            elif name == 'rng':
                out[name] = rng._require()
            elif name == 'state':
                out[name] = formed_input
            elif name in formed_input:
                out[name] = formed_input[name]
        return frozendict(out)

    if primes:
        def impl(definition, param, formed_input, input, rng):
            call = dict(arguments(definition, param, formed_input, rng))
            call['input'] = input
            return _keys_only(fn(**call), f'{definition.name}.init')
    else:
        def impl(definition, param, formed_input, rng):
            return _keys_only(
                fn(**arguments(definition, param, formed_input, rng)),
                f'{definition.name}.init')

    return InitCall(
        impl=impl,
        form=CallForm.from_values(declaration),
        takes_rng=takes_rng,
        requires_input=primes,
        reads_def='node' in signature,
    )


_RESERVED_METHOD_NAMES = frozenset({
    'apply', 'init', 'scan', 'parameterize', 'bind', 'specialize',
    'initialize', 'reset', 'with_input', 'name', 'generic', 'parametric',
    'cyclic', 'bound', 'resolved', 'node', 'pnode', 'param', 'state',
    'contract', 'members',
    'input', 'input_spec', 'input_shape',
    'describe', 'tree_view', 'summary', 'statics_by_path',
})


def _check_methods(methods) -> frozendict:
    methods = dict(methods)
    collisions = set(methods) & _RESERVED_METHOD_NAMES
    if collisions:
        raise TypeError(
            f'method names shadow Node attributes: {sorted(collisions)}')
    for name, fn in methods.items():
        _check_method_signature(name, fn)
    return frozendict(methods)


_CHANNEL_RANK = {'node': 0, 'param': 1, 'state': 2, 'rng': 3}


def _check_method_signature(name: str, fn: Callable) -> None:
    names = list(_signature(fn, f'method {name!r}'))
    if 'self' in names:
        raise TypeError(f'method {name!r}: use param, not self')
    channels = [item for item in names if item in _METHOD_CHANNELS]
    if names[:len(channels)] != channels:
        raise TypeError(f'method {name!r}: injected channels come first')
    ranks = [_CHANNEL_RANK[channel] for channel in channels]
    if ranks != sorted(ranks):
        raise TypeError(
            f'method {name!r}: channel order is node, param, state, rng')
