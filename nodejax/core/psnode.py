"""The parameter-and-state-bound public view."""

from __future__ import annotations

from typing import Any

import jax

from nodejax.core.binding import _UNSET, _bind_method, _bind_public_call
from nodejax.core.definition import Def
from nodejax.core.node import BaseNode
from nodejax.core.pnode import PNode


class PSNode(BaseNode):
    """A definition with parameter and live state trees bound as JAX data."""

    bound = True
    state_bound = True

    def __init__(self, definition: Def, param, state):
        super().__init__(definition)
        self.param = self.contract._sparse_param(param)
        self.state = self.contract._sparse_state(state)

    def _binding_items(self):
        return (('param', self.param), ('state', self.state))

    def _with_definition(self, definition: Def) -> 'PSNode':
        return PSNode(definition, self.param, self.state)

    def _transfer_bindings(
            self, target: BaseNode, preserves=('param', 'state'),
            strict: bool = False, operation: str = 'operation') -> BaseNode:
        preserved = frozenset(preserves)
        if strict and 'state' not in preserved:
            raise TypeError(
                f"'{self.name}' is state-bound; {operation} does not "
                'preserve state')
        if strict and self.parametric and 'param' not in preserved:
            raise TypeError(
                f"'{self.name}' is parameter-bound; {operation} does not "
                'preserve parameters')
        if 'state' in preserved:
            return PSNode(target._def, self.param, self.state)
        if 'param' in preserved:
            return PNode(target._def, self.param)
        return target

    def _internalized_state(self) -> tuple[BaseNode, Any]:
        """A state-bound view hands its state to a run and keeps its params."""
        return self.pnode, self.state

    @property
    def pnode(self) -> PNode:
        return PNode(self._def, self.param)

    @property
    def param_spec(self):
        """The shapes of the parameters this view holds."""
        from nodejax.core.spec import spec_of
        return spec_of(self.param)

    @property
    def state_spec(self):
        """The shapes of the state this view holds."""
        from nodejax.core.spec import spec_of
        return spec_of(self.state)

    @property
    def output_spec(self):
        return self.contract.output_spec_from(self.param_spec, self.state_spec)

    def reset(self, *args, **fields) -> 'PSNode':
        return self.pnode.initialize(*args, **fields)

    def apply(self, *args, **fields):
        formed, rng = _bind_public_call(self, args, fields)
        state, output = self.contract.apply(
            self.param, self.state, formed, rng)
        return PSNode(self._def, self.param, state), output

    def __call__(self, *args, **fields):
        return self.apply(*args, **fields)

    def scan(self, *args, **fields):
        from nodejax.transforms.iteration.scan import scan

        runner = scan(self.pnode)
        formed, rng = _bind_public_call(runner, args, fields)
        return _session_scan(self, formed, rng)

    def bind(self, param=_UNSET, *, state=_UNSET):
        return PSNode(
            self._def,
            self.param if param is _UNSET else param,
            self.state if state is _UNSET else state,
        )

    def __getattr__(self, name: str):
        if name in self._def.methods:
            from nodejax.core.author_view import AuthorNode
            # Methods see the dense slices an authored apply sees.
            return _bind_method(
                self._def.methods[name],
                param=lambda: self.contract._dense_param(self.param),
                state=lambda: self.contract._dense_state(self.state),
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
            f'PSNode {self.name!r} has no attribute {name!r}; '
            f'values live under .param and .state{available}')

    def _member(self, name: str) -> 'PSNode':
        """The member view with its parameter and state slices."""
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
        if member.cyclic and not self._def.layout.destructurable_state:
            raise AttributeError(
                f'{self.name!r} maps state over an axis; '
                'a member has no single state slice')
        if self._def.layout.transparent_member == name:
            param, state = self.param, self.state
        else:
            param = getattr(self.param, name) if member.parametric else ()
            state = getattr(self.state, name) if member.cyclic else ()
        return PSNode(member, param, state)

    def __rshift__(self, other):
        from nodejax.core.compose import _compose
        return _compose(self, other)

    def __repr__(self) -> str:
        return (f'PSNode({self.name}, param={self.param!r}, '
                f'state={self.state!r})')


@jax.jit
def _session_scan(session: PSNode, inputs, rng):
    from nodejax.transforms.iteration.scan import scan

    runner = scan(session.pnode)
    state, outputs = runner.contract.apply(
        runner.param, session.state, inputs, rng)
    return PSNode(session._def, session.param, state), outputs


def _flatten(node: PSNode):
    return (node.param, node.state), node._def


def _flatten_with_keys(node: PSNode):
    return (
        (jax.tree_util.GetAttrKey('param'), node.param),
        (jax.tree_util.GetAttrKey('state'), node.state),
    ), node._def


def _unflatten(definition: Def, children):
    param, state = children
    return PSNode(definition, param, state)


jax.tree_util.register_pytree_with_keys(
    PSNode, _flatten_with_keys, _unflatten, _flatten)
