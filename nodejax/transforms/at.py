from __future__ import annotations

from nodejax.core import Node, NodeDef, _input_or_none, _resolve
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.spec import materialize
from nodejax.transforms.common import _split, _rewrap


@_over_generic
def at(node: NodeDef | Node, field: str) -> NodeDef | Node:
    """Route a node onto one field of a Struct input: the output is the
    input Struct with `field` replaced by the node's output, every
    other field passed through untouched. Pipes chain whole signals;
    at() lets a chain act on one strand of a structured signal while
    the rest ride alongside.

    Type-preserving and transparent: param and state are the inner
    node's own, an offered init input is projected to the field, and
    the wrapper keeps the inner node's name, so pipe member keys and
    state paths read as if the node were placed directly."""
    nd, param = _split(node)

    def init_fn(ndef, p, state_input=Struct(), input=None):
        carry = input if input is not None else _input_or_none(ndef)
        if carry is None:
            return nd.build_state(p, state_input)
        fld = materialize(carry)[field]
        return _resolve(nd, fld).build_state(p, state_input,
                                             input=fld)

    def apply_fn(p, s, i):
        s2, out = nd.apply_fn(p, s, i[field])
        return s2, i.replace(**{field: out})

    out = NodeDef(nd.name, nd._param_impl, init_fn, apply_fn, nd.parametric, nd.cyclic,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.init_reads_shape,
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None)
    return _rewrap(out, param)
