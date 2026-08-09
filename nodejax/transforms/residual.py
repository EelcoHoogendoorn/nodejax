from __future__ import annotations

from typing import Any
from nodejax.compose import wrapper
from nodejax.core import Node, NodeDef
from nodejax.generic import GenericDef


def residual(nd: GenericDef | NodeDef | Node) -> GenericDef | NodeDef | Node:
    """x + f(x): the skip connection around any shape-preserving node."""
    return wrapper(lambda self, input: input + self(input), nd, name=f'res({nd.name})')
