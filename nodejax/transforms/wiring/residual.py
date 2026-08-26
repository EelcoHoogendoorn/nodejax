from __future__ import annotations

from typing import Any
from nodejax.core.wrapper import (Wrapper)
from nodejax.core.ambient import node
from nodejax.core.node import (Node)
from nodejax.core.pnode import (PNode)


@node
def residual(body: Node) -> Node:
    """x + f(x): the skip connection around any shape-preserving node."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name=f'res({body.name})')
