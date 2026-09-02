"""The parameter-bound public view."""

from __future__ import annotations

from typing import Any

import jax

from nodejax.core.binding import (
    _UNSET, _as_bundle, _bind_method, _bind_public_call, _bind_rng,
    validate_state_input,
)
from nodejax.core.definition import Def
from nodejax.core.node import BaseNode
from nodejax.struct import Struct


class PNode(BaseNode):
    """A definition with its parameter tree bound as JAX data."""

    bound = True

    def __init__(self, definition: Def, param):
        super().__init__(definition)
        self.param = self.contract._sparse_param(param)

    def _binding_items(self):
        return (('param', self.param),)

    def _with_definition(self, definition: Def) -> 'PNode':
        return PNode(definition, self.param)

    def _transfer_bindings(
            self, target: BaseNode, preserves=('param', 'state'),
            strict: bool = False, operation: str = 'operation') -> BaseNode:
        preserved = frozenset(preserves)
        if strict and not self.parametric:
            return target
        if strict and 'param' not in preserved:
            raise TypeError(
                f"'{self.name}' is parameter-bound; {operation} does not "
                'preserve parameters')
        if 'param' in preserved:
            return PNode(target._def, self.param)
        return target

    @property
    def pnode(self) -> 'PNode':
        return self

    def init(self, state_input: Any = _UNSET, /, *,
             input=_UNSET, **fields):
        key = fields.pop('rng', _UNSET)
        if state_input is not _UNSET and fields:
            raise TypeError('pass one state bundle or loose fields')
        bundle = (_as_bundle(fields) if state_input is _UNSET
                  else state_input)
        if type(bundle) is not Struct:
            raise TypeError('init expects a Struct or loose fields')
        validate_state_input(self, bundle)
        if not self.cyclic:
            # A stateless node initializes to the empty state. A priming
            # input is vacuous data and accepted; a claimed state field
            # already raised above, and an unconsumed key stays a leak.
            _bind_rng(False, key, f'{self.name}: init')
            return ()
        rng = _bind_rng(
            self.contract.init_takes_rng, key, f'{self.name}: init')
        if input is _UNSET:
            state = self.contract.init(self.param, bundle, rng)
        else:
            state = self.contract.prime(self.param, bundle, input, rng)
        return self.contract._sparse_state(state)

    def initialize(self, state_input: Any = _UNSET, /, *,
                   input=_UNSET, **fields):
        from nodejax.core.psnode import PSNode
        state = self.init(state_input, input=input, **fields)
        return PSNode(self._def, self.param, state)

    def apply(self, *args, **fields):
        if self.cyclic:
            if not args:
                raise TypeError(
                    f'{self.name} is cyclic; pass state positionally')
            formed, rng = _bind_public_call(self, args[1:], fields)
            state, output = self.contract.apply(
                self.param, args[0], formed, rng)
            return self.contract._sparse_state(state), output
        formed, rng = _bind_public_call(self, args, fields)
        _, output = self.contract.apply(self.param, (), formed, rng)
        return output

    def __call__(self, *args, **fields):
        return self.apply(*args, **fields)

    def bind(self, param=_UNSET, *, state=_UNSET):
        actual_param = self.param if param is _UNSET else param
        if state is _UNSET:
            return PNode(self._def, actual_param)
        from nodejax.core.psnode import PSNode
        return PSNode(self._def, actual_param, state)

    def __getattr__(self, name: str):
        if name in self._def.methods:
            from nodejax.core.author_view import AuthorNode
            # Methods see the dense slices an authored apply sees.
            return _bind_method(
                self._def.methods[name],
                param=lambda: self.contract._dense_param(self.param),
                node=lambda: AuthorNode(self._def),
            )
        if name in self._def.members:
            return self._member(name)
        transparent = self._def.layout.transparent_member
        if transparent is not None:
            # A transparent wrapper adds no level to the tree, so its
            # member's attributes are reached without naming it.
            return getattr(self._member(transparent), name)
        methods = tuple(self._def.methods)
        available = f'; authored methods: {methods}' if methods else ''
        raise AttributeError(
            f'PNode {self.name!r} has no attribute {name!r}; '
            f'values live under .param{available}')

    def _member(self, name: str) -> 'PNode':
        """The member view with its parameter slice."""
        if not self._def.layout.destructurable_param:
            raise AttributeError(
                f'{self.name!r} maps parameters over an axis; '
                'a member has no single parameter slice')
        member = getattr(self._def.members, name)
        param_members = self._def.layout.param_members
        if (member.parametric and param_members is not None
                and name not in param_members):
            raise TypeError(
                f"member {name!r} has no slot in {self.name!r}'s "
                'parameter tree')
        if self._def.layout.transparent_member == name:
            param = self.param
        else:
            param = getattr(self.param, name) if member.parametric else ()
        return PNode(member, param)

    def __repr__(self) -> str:
        return f'PNode({self.name}, param={self.param!r})'


class _ParamHop:
    def __str__(self):
        return ''

    def __repr__(self):
        return '<param>'


_PARAM_HOP = _ParamHop()


def _flatten(node: PNode):
    return (node.param,), node._def


def _flatten_with_keys(node: PNode):
    return ((_PARAM_HOP, node.param),), node._def


def _unflatten(definition: Def, children):
    param, = children
    return PNode(definition, param)


jax.tree_util.register_pytree_with_keys(
    PNode, _flatten_with_keys, _unflatten, _flatten)
