"""Unbound public view and shared `BaseNode` behavior over a `Def`.

Every completed NodeJAX program is an immutable `Def`. `BaseNode` exposes the
structural inspection, static specialization, and input or parameter binding
methods shared by all public views, while `Node` represents the unbound view
before parameter values are attached (`PNode` and `PSNode` hold the bound
views). Non-parametric definitions skip `Node` at construction time and return
an empty-parameter `PNode` directly so stateless pure Nodes can be called
immediately.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, TYPE_CHECKING

from nodejax.core.binding import (
    _UNSET, _as_bundle, _bind_rng, validate_param_input,
)
from nodejax.core.contract import Contract
from nodejax.core.definition import Def
from nodejax.core.types import PyTree
from nodejax.frozendict import frozendict
from nodejax.struct import Struct

if TYPE_CHECKING:
    from nodejax.core.pnode import PNode
    from nodejax.core.psnode import PSNode


def _view(definition: Def) -> Node | PNode:
    if definition.parametric:
        return Node(definition)
    from nodejax.core.pnode import PNode
    return PNode(definition, ())


class BaseNode:
    """Provides the shared structural and binding interface over a completed `Def`."""

    bound = False
    state_bound = False

    def __init__(self, definition: Def) -> None:
        if type(definition) is not Def:
            raise TypeError(f'{type(self).__name__} expects a Def')
        self._def = definition

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def tags(self) -> frozenset[str]:
        return self._def.tags

    @cached_property
    def members(self) -> Struct:
        """Exposes direct child definitions as a `Struct` of unbound `Node` views."""
        return Struct(**{
            name: Node(definition)
            for name, definition in self._def.members.__items__
        })

    @property
    def contract(self) -> Contract:
        """Exposes the low-level dense execution `Contract` of the underlying `Def`."""
        return self._def.contract

    @property
    def node(self) -> Node:
        """Drops any bound parameter or state trees and returns the unbound `Node`."""
        return Node(self._def)

    @property
    def generic(self) -> bool:
        return False

    @property
    def parametric(self) -> bool:
        return self._def.parametric

    @property
    def cyclic(self) -> bool:
        return self._def.cyclic

    def _binding_items(self) -> tuple[tuple[str, PyTree], ...]:
        return ()

    def _with_definition(self, definition: Def) -> BaseNode:
        return Node(definition)

    def _transfer_bindings(
            self, target: BaseNode,
            preserves: tuple[str, ...] = ('param', 'state'),
            strict: bool = False, operation: str = 'operation') -> BaseNode:
        return target

    def with_input(self, input_spec: PyTree = _UNSET, /, *,
                   bundle: Struct = _UNSET) -> Node | PNode:
        """Resolves input-dependent shapes against an example or spec and discards bound trees.

        A positional argument supplies a single input value or `ArraySpec`,
        whereas `bundle=` supplies a multi-field `Struct` matching a keyword
        `apply` call.
        """
        if input_spec is not _UNSET and bundle is not _UNSET:
            raise TypeError('pass one input wire or bundle=, not both')
        if input_spec is _UNSET and bundle is _UNSET:
            raise TypeError('with_input requires one input wire or bundle=')
        if bundle is _UNSET:
            return self.contract.with_input(input_spec)
        return _view(self.contract._resolve_def(
            bundle, replace_evidence=True, bundled=True))

    def specialize(self, **overrides: PyTree) -> Node:
        """Replays recorded construction with dotted-path static overrides and drops bindings."""
        from nodejax.core.construction import specialize
        return Node(specialize(self._def, overrides))

    def parameterize(self, param_input: PyTree = _UNSET, /,
                     **fields: PyTree) -> PNode:
        """Initializes parameter trees from explicit overrides and `rng`, returning a `PNode`."""
        key = fields.pop('rng', _UNSET)
        if param_input is not _UNSET and fields:
            raise TypeError('pass one parameter bundle or loose fields')
        bundle = (_as_bundle(fields) if param_input is _UNSET
                  else param_input)
        if type(bundle) is not Struct:
            raise TypeError('parameterize expects a Struct or loose fields')
        validate_param_input(self, bundle)
        rng = _bind_rng(
            self.contract.param_takes_rng, key,
            f'{self.name}: parameterize')
        from nodejax.core.pnode import PNode
        return PNode(self._def, self.contract.param(bundle, rng))

    def describe(self) -> str:
        from nodejax.core.printing import describe
        return describe(self)

    def tree_view(self, *, max_depth: int | None = None,
                  unicode: bool = True) -> str:
        from nodejax.core.printing import tree_view
        return tree_view(self, max_depth=max_depth, unicode=unicode)

    def summary(self, *, max_depth: int | None = None,
                print_fn: Callable | None = None) -> str:
        from nodejax.core.printing import summary
        return summary(self, max_depth=max_depth, print_fn=print_fn)

    def statics_by_path(self) -> frozendict:
        from nodejax.core.printing import statics_by_path
        return statics_by_path(self)

    def __rshift__(self, other: PyTree) -> BaseNode:
        from nodejax.core.compose import _compose
        return _compose(self, other)

    def __repr__(self) -> str:
        roles = 'P' * self.parametric + 'C' * self.cyclic
        suffix = f':{roles}' if roles else ''
        return f'{type(self).__name__}({self.name}{suffix})'


def _is_node(value: PyTree) -> bool:
    return issubclass(type(value), BaseNode)


class Node(BaseNode):
    """Exposes a parametric `Def` prior to parameter initialization.

    Calling a `Node` invokes `parameterize(...)` and returns a `PNode`. Direct
    `bind`, `init`, and `apply` methods also accept an explicit `param` tree
    without constructing a persistent `PNode` first.
    """

    @property
    def node(self) -> Node:
        return self

    def __getattr__(self, name: str):
        if name in self._def.methods:
            return self._def.methods[name]
        if name in self._def.members:
            return Node(getattr(self._def.members, name))
        transparent = self._def.layout.transparent_member
        if transparent is not None:
            return getattr(Node(getattr(self._def.members, transparent)), name)
        raise AttributeError(f'Node {self.name!r} has no attribute {name!r}')

    def bind(self, param=(), *, state=_UNSET) -> PNode | PSNode:
        """Attaches an existing `param` tree (and optional `state` tree) to produce `PNode` or `PSNode`."""
        if state is _UNSET:
            from nodejax.core.pnode import PNode
            return PNode(self._def, param)
        from nodejax.core.psnode import PSNode
        return PSNode(self._def, param, state)

    def apply(self, param, *args: PyTree, **fields: PyTree) -> PyTree:
        """Evaluates one step against an explicit `param` tree (and positional `state` when cyclic)."""
        from nodejax.core.pnode import PNode
        return PNode(self._def, param).apply(*args, **fields)

    def init(self, param, state_input: PyTree = _UNSET, /, *,
             input=_UNSET, **fields: PyTree) -> PyTree:
        """Computes a fresh sparse `state` tree using an explicit `param` tree."""
        from nodejax.core.pnode import PNode
        return PNode(self._def, param).init(
            state_input, input=input, **fields)

    def __call__(self, *args: PyTree, **fields: PyTree) -> PNode:
        return self.parameterize(*args, **fields)
