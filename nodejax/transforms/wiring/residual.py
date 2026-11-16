"""Additive skip-connection wrapper (`residual`)."""

from __future__ import annotations

from nodejax.core.ambient import node
from nodejax.core.node import BaseNode
from nodejax.core.wrapper import Wrapper


@node
def residual(body: BaseNode) -> BaseNode:
    """Adds `input` to `body(input)` around a shape-preserving `body` Node while retaining `body`'s input spec."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    spec = body.contract.input_spec
    return wrapped(
        apply, name=f'res({body.name})',
        input_spec=None if spec is None else body.contract.intake(spec))
