"""Read-only definition metadata exposed to authored functions as `node`."""

from __future__ import annotations

from typing import Callable

import jax

from nodejax.core.binding import _spec_resolved
from nodejax.core.definition import Def
from nodejax.core.types import PyTree
from nodejax.struct import Struct


class AuthorNode:
    """Exposes resolved input specs, member views, and custom methods on `node`.

    Authored `param`, `init`, `apply`, and method functions receive `AuthorNode`
    when they declare `node` in their signature, giving them read-only access to
    static definition facts without exposing mutable or contract-level internals.
    """

    def __init__(self, definition: Def) -> None:
        self._def = definition

    def __getattr__(self, name: str) -> Callable:
        if name in self._def.methods:
            return self._def.methods[name]
        raise AttributeError(
            f"authored Node {self._def.name!r} has no attribute {name!r}")

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def members(self) -> Struct:
        """Returns a `Struct` of unbound Node views for each declared member."""
        from nodejax.core.node import _view
        return Struct(**{
            name: _view(member) for name, member in self._def.members.__items__
        })

    @property
    def input_spec(self) -> PyTree:
        """Returns the resolved `apply` input spec unpacked to the authored wire shape."""
        spec = self._def.contract.input_spec
        if not _spec_resolved(spec):
            raise TypeError(
                f'{self._def.name}: authored code reads node.input_spec but no '
                'input shape has been resolved')
        return self._def.contract.intake(spec)

    @property
    def input_shape(self) -> PyTree:
        """Returns a PyTree of leaf `.shape` tuples matching `self.input`."""
        return jax.tree.map(lambda leaf: leaf.shape, self.input)

    @property
    def input(self) -> PyTree:
        """Materializes zero-valued array exemplars matching `self.input_spec`."""
        from nodejax.core.spec import materialize
        return materialize(self.input_spec)
