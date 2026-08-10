from __future__ import annotations

import jax

from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.transforms.common import (_over_bound, _mapped_init_fn,
                                       _scanned_apply_fn, _transform_def)


@_over_generic
@_over_bound
def repeat(node_def: NodeDef, n: int) -> NodeDef:
    """Apply the SAME node n times in sequence: weight-tied depth.

    Where stack gives every layer its own params (a leading layer axis),
    repeat threads one set of params through n applications — iterated
    refinement. Param meaning is unchanged, so bound Nodes rebind. Cyclic
    nodes keep one state slot PER position (state gains a leading axis of
    n): tied weights, untied state. Output must be input-shaped — the same
    carry contract as stack.
    """
    count = n

    def apply_fn(nd, param, state, input):
        return _scanned_apply_fn(node_def, param, state, input,
                           stacked_param=False, length=count)

    return _transform_def(
        node_def,
        name=f'repeat({node_def.name})',
        init_fn=_mapped_init_fn(node_def, count, stacked=False),
        apply_fn=apply_fn,
        rebuild=lambda d: repeat(d, n=count),
    )
