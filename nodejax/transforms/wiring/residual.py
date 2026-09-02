from __future__ import annotations

from typing import Any
from nodejax.core.wrapper import (Wrapper)
from nodejax.core.ambient import node
from nodejax.core.node import (Node)
from nodejax.core.pnode import (PNode)


@node
def residual(body: Node) -> Node:
    """x + f(x): the skip connection around any shape-preserving node.

    The body takes exactly what the residual is handed, so a resolved
    body's input is the residual's own declared input."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    spec = body.contract.input_spec
    return wrapped(
        apply, name=f'res({body.name})',
        input_spec=None if spec is None else body.contract.intake(spec))
