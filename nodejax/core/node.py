"""Public definition views.

The complete program is a :class:`nodejax.core.definition.Def`.  ``Node`` is its
unbound public view; the bound sibling views live in ``pnode`` and ``psnode``.
No contract storage or structural reconstruction machinery belongs here.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable

from nodejax.core.binding import (
    _UNSET, _as_bundle, _bind_rng, validate_param_input,
)
from nodejax.core.contract import Contract
from nodejax.core.definition import Def
from nodejax.frozendict import frozendict
from nodejax.struct import Struct


def _view(definition: Def):
    """Return the ordinary public view appropriate for one definition."""
    if definition.parametric:
        return Node(definition)
    from nodejax.core.pnode import PNode
    return PNode(definition, ())


class BaseNode:
    """Small common public view over one complete definition."""

    bound = False
    state_bound = False

    def __init__(self, definition: Def):
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
        return Struct(**{
            name: Node(definition)
            for name, definition in self._def.members.__items__
        })

    @property
    def contract(self) -> Contract:
        return self._def.contract

    @property
    def node(self) -> 'Node':
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

    @property
    def param_spec(self):
        """The parameter tree as shapes, from the resolved input spec."""
        return self.contract.param_spec

    @property
    def state_spec(self):
        """The state tree as shapes, from the resolved input spec and the
        parameter shapes."""
        return self.contract.state_spec

    @property
    def output_spec(self):
        """The apply output as shapes, from the resolved input spec."""
        return self.contract.output_spec

    def _binding_items(self) -> tuple[tuple[str, Any], ...]:
        """Runtime bindings carried by this view, in ladder order."""
        return ()

    def _with_definition(self, definition: Def) -> 'BaseNode':
        """Return the unbound view over a replacement definition."""
        return Node(definition)

    def _transfer_bindings(
            self, target: 'BaseNode', preserves=('param', 'state'),
            strict: bool = False, operation: str = 'operation') -> 'BaseNode':
        """An unbound view contributes no bindings to ``target``."""
        return target

    def _internalized_state(self) -> tuple['BaseNode', Any]:
        """The view without its state binding, and the state a run may
        start from; an unbound view has none."""
        return self, None

    def with_input(self, input_spec: Any = _UNSET, /, *,
                   bundle: Struct = _UNSET) -> 'Node':
        """Re-enter input binding and discard later runtime bindings.

        A positional value is one computed wire. ``bundle=`` supplies an
        already formed multi-field call, matching the public apply spelling.
        """
        if input_spec is not _UNSET and bundle is not _UNSET:
            raise TypeError('pass one input wire or bundle=, not both')
        if input_spec is _UNSET and bundle is _UNSET:
            raise TypeError('with_input requires one input wire or bundle=')
        if bundle is _UNSET:
            return self.contract.with_input(input_spec)
        return _view(self.contract._resolve_def(
            bundle, replace_evidence=True, bundled=True))

    def specialize(self, **overrides: Any) -> 'Node':
        """Replay canonical construction with accumulated static overrides."""
        from nodejax.core.construction import specialize
        return Node(specialize(self._def, overrides))

    def parameterize(self, param_input: Any = _UNSET, /,
                     **fields) -> 'PNode':
        """Re-enter parameter binding and discard any bound state."""
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

    def __rshift__(self, other):
        from nodejax.core.compose import _compose
        return _compose(self, other)

    def __repr__(self) -> str:
        roles = 'P' * self.parametric + 'C' * self.cyclic
        suffix = f':{roles}' if roles else ''
        return f'{type(self).__name__}({self.name}{suffix})'


def _is_node(value: Any) -> bool:
    """Whether ``value`` implements the concrete BaseNode hierarchy."""
    return issubclass(type(value), BaseNode)


class Node(BaseNode):
    """Unbound public definition view."""

    @property
    def node(self) -> 'Node':
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

    def bind(self, param=(), *, state=_UNSET):
        if state is _UNSET:
            from nodejax.core.pnode import PNode
            return PNode(self._def, param)
        from nodejax.core.psnode import PSNode
        return PSNode(self._def, param, state)

    def apply(self, param, *args, **fields):
        from nodejax.core.pnode import PNode
        return PNode(self._def, param).apply(*args, **fields)

    def init(self, param, state_input: Any = _UNSET, /, *,
             input=_UNSET, **fields):
        from nodejax.core.pnode import PNode
        return PNode(self._def, param).init(
            state_input, input=input, **fields)

    def __call__(self, *args, **fields):
        return self.parameterize(*args, **fields)
