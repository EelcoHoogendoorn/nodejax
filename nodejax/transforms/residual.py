from __future__ import annotations

from nodejax.core import Node, NodeDef
from nodejax.generic import _over_generic
from nodejax.transforms.common import _split, _rewrap


@_over_generic
def residual(node: NodeDef | Node) -> NodeDef | Node:
    """x + f(x): the skip connection as a transform, for any
    shape-preserving node — state, if any, rides through untouched.
    Param meaning is unchanged, so bound Nodes rebind."""
    nd, param = _split(node)

    def apply_fn(p, s, i):
        new_state, out = nd.apply_fn(p, s, i)
        return new_state, i + out

    out = NodeDef(f'res({nd.name})', nd._param_impl, nd._init_impl, apply_fn,
                  nd.parametric, nd.cyclic, apply_input_spec=nd.apply_input_spec,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.init_reads_shape,
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None)
    return _rewrap(out, param)
