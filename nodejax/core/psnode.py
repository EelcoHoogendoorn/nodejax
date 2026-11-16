"""Parameter-and-state-bound public view (`PSNode`) registered as a JAX PyTree.

A `PSNode` binds both sparse `param` and sparse `state` trees as PyTree children
under `.param` and `.state` attribute keys while holding `Def` as static
metadata. Calling `psnode(x)` advances one step and returns `(next_psnode,
output)` without mutating the receiver, making `PSNode` directly usable as a
`jax.lax.scan` carry or interactive rollout handle.
"""

from __future__ import annotations

import jax

from nodejax.core.binding import _UNSET, _bind_method, _bind_public_call
from nodejax.core.definition import Def
from nodejax.core.node import BaseNode
from nodejax.core.pnode import PNode
from nodejax.core.types import PyTree


class PSNode(BaseNode):
    """Binds sparse `param` and `state` trees to a `Def` as JAX PyTree data."""

    bound = True
    state_bound = True

    def __init__(self, definition: Def, param, state) -> None:
        super().__init__(definition)
        self.param = self.contract._sparse_param(param)
        self.state = self.contract._sparse_state(state)

    def _binding_items(self) -> tuple[tuple[str, PyTree], ...]:
        return (('param', self.param), ('state', self.state))

    def _with_definition(self, definition: Def) -> PSNode:
        return PSNode(definition, self.param, self.state)

    def _transfer_bindings(
            self, target: BaseNode,
            preserves: tuple[str, ...] = ('param', 'state'),
            strict: bool = False, operation: str = 'operation') -> BaseNode:
        preserved = frozenset(preserves)
        if strict and 'state' not in preserved and self.cyclic:
            raise TypeError(
                f"'{self.name}' is state-bound; {operation} does not "
                'preserve state')
        if strict and self.parametric and 'param' not in preserved:
            raise TypeError(
                f"'{self.name}' is parameter-bound; {operation} does not "
                'preserve parameters')
        if 'state' in preserved and self.cyclic:
            return PSNode(target._def, self.param, self.state)
        if 'param' in preserved:
            return PNode(target._def, self.param)
        return target

    @property
    def pnode(self) -> PNode:
        """Drops the bound `state` tree and returns the underlying `PNode`."""
        return PNode(self._def, self.param)

    def parameterize(self, param_input: PyTree = _UNSET, /,
                     **fields: PyTree) -> PNode:
        """Returns `self.pnode` when called with no overrides, or re-parameterizes `self._def`."""
        if param_input is _UNSET and not fields:
            return self.pnode
        return super().parameterize(param_input, **fields)

    def reset(self, *args: PyTree, **fields: PyTree) -> PSNode:
        """Re-runs state initialization while keeping `param` fixed, returning a fresh `PSNode`."""
        return self.pnode.initialize(*args, **fields)

    def apply(self, *args: PyTree, **fields: PyTree) -> tuple[PSNode, PyTree]:
        """Advances one step from `self.state` and returns `(next_psnode, output)`."""
        formed, rng = _bind_public_call(self, args, fields)
        state, output = self.contract.apply(
            self.param, self.state, formed, rng)
        return PSNode(self._def, self.param, state), output

    def __call__(self, *args: PyTree, **fields: PyTree) -> tuple[PSNode, PyTree]:
        return self.apply(*args, **fields)

    def scan(self, *args: PyTree, **fields: PyTree) -> tuple[PSNode, PyTree]:
        """Scans an input sequence along leading axis 0 under `jax.jit`, returning `(next_psnode, outputs)`."""
        from nodejax.transforms.iteration.scan import scan

        runner = scan(self.pnode)
        formed, rng = _bind_public_call(runner, args, fields)
        return _session_scan(self, formed, rng)

    def bind(self, param=_UNSET, *, state=_UNSET) -> PSNode:
        """Replaces `param`, `state`, or both and returns the updated `PSNode`."""
        return PSNode(
            self._def,
            self.param if param is _UNSET else param,
            self.state if state is _UNSET else state,
        )

    def __getattr__(self, name: str):
        if name in self._def.methods:
            from nodejax.core.author_view import AuthorNode
            # Authored `@node.method` functions receive the dense parameter and state trees.
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
            # Transparent wrappers forward member lookup without adding a tree level.
            return getattr(self._member(transparent), name)
        methods = tuple(self._def.methods)
        available = f'; authored methods: {methods}' if methods else ''
        raise AttributeError(
            f'PSNode {self.name!r} has no attribute {name!r}; '
            f'values live under .param and .state{available}')

    def _member(self, name: str) -> PSNode:
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
            param, state = self.param, (self.state if member.cyclic else ())
        else:
            param = getattr(self.param, name) if member.parametric else ()
            state = getattr(self.state, name) if member.cyclic else ()
        return PSNode(member, param, state)

    def __rshift__(self, other: PyTree) -> BaseNode:
        from nodejax.core.compose import _compose
        return _compose(self, other)

    def __repr__(self) -> str:
        return (f'PSNode({self.name}, param={self.param!r}, '
                f'state={self.state!r})')


@jax.jit
def _session_scan(bound: PSNode, inputs: PyTree, rng) -> tuple[PSNode, PyTree]:
    from nodejax.transforms.iteration.scan import scan

    runner = scan(bound.pnode)
    state, outputs = runner.contract.apply(
        runner.param, bound.state, inputs, rng)
    return PSNode(bound._def, bound.param, state), outputs


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
