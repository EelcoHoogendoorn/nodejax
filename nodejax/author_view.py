"""Restricted definition view injected into authored functions."""

from __future__ import annotations

import jax

from nodejax.binding import _spec_resolved
from nodejax.definition import Def


class AuthorNode:
    """Static facts safe for authored bodies to observe.

    This is intentionally not a public Node and not a Contract.  New fields
    belong here only after an authored function demonstrates a concrete need.
    """

    def __init__(self, definition: Def):
        self._def = definition

    def __getattr__(self, name: str):
        """Expose the current definition's methods without binding values."""
        if name in self._def.methods:
            return self._def.methods[name]
        raise AttributeError(
            f"authored Node {self._def.name!r} has no attribute {name!r}")

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def input_spec(self):
        """Resolved input specification projected to the authored call."""
        spec = self._def.contract.input_spec
        if not _spec_resolved(spec):
            raise TypeError(
                f'{self._def.name}: authored code reads node.input_spec but no '
                'input shape has been resolved')
        return self._def.contract.intake(spec)

    @property
    def input_shape(self):
        """Input shapes projected to the authored call."""
        return jax.tree.map(lambda leaf: leaf.shape, self.input)

    @property
    def input(self):
        """Materialized input evidence projected to the authored call."""
        from nodejax.spec import materialize
        return materialize(self.input_spec)
