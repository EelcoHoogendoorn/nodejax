"""Fan-out and leaf-wise summation (`sum_junction`) across parallel branches."""

from __future__ import annotations

import jax

from nodejax.core.composite import Composite
from nodejax.core.generic import Generic
from nodejax.core.node import BaseNode


def sum_junction(**members: BaseNode | Generic) -> BaseNode | Generic:
    """Broadcasts one shared `input` to all named `members` and sums their output PyTrees leaf-wise."""
    if not members:
        raise TypeError('sum_junction needs at least one member')

    def add(self, input):
        outputs = [run(input) for _, run in self.__items__]
        return jax.tree.map(lambda *terms: sum(terms), *outputs)

    return Composite(**members)(
        add, name='(' + ' + '.join(members) + ')')
