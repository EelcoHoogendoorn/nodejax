from __future__ import annotations

import jax

from nodejax.core.composite import Composite


def sum_junction(**members):
    """Broadcast one input to named members and add their outputs."""
    if not members:
        raise TypeError('sum_junction needs at least one member')

    def add(self, input):
        outputs = [run(input) for _, run in self.__items__]
        return jax.tree.map(lambda *terms: sum(terms), *outputs)

    return Composite(**members)(
        add, name='(' + ' + '.join(members) + ')')
