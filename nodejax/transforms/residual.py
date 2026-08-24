from __future__ import annotations

from typing import Any
from nodejax.wrapper import (Wrapper)
from nodejax.ambient import node
from nodejax.node import (Node)
from nodejax.pnode import (PNode)


@node
def residual(body: Node | PNode) -> Node | PNode:
    """x + f(x): the skip connection around any shape-preserving node."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name=f'res({body.name})')
