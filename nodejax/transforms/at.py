from __future__ import annotations

from nodejax.core import Node, NodeDef, _resolve
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.spec import materialize
from nodejax.transforms.common import _over_bound, _transform_def


@_over_generic
@_over_bound
def at(node_def: NodeDef, field: str) -> NodeDef:
    """Route a node onto one field of a Struct input: the output is the
    input Struct with `field` replaced by the node's output, every
    other field passed through untouched. Pipes chain whole signals;
    at() lets a chain act on one strand of a structured signal while
    the rest ride alongside.

    Type-preserving and transparent: param and state are the inner
    node's own, an offered init input is projected to the field, and
    the wrapper keeps the inner node's name, so pipe member keys and
    state paths read as if the node were placed directly."""

    def init_fn(ndef, p, state_input=Struct(), input=None):
        carry = input if input is not None else (ndef.input if ndef.resolved else None)
        if carry is None:
            return node_def.build_state(p, state_input)
        fld = materialize(carry)[field]
        return _resolve(node_def, fld).build_state(p, state_input,
                                             input=fld)

    def apply_fn(nd, p, s, i):
        s2, out = node_def.apply_fn(p, s, i[field])
        return s2, i.replace(**{field: out})

    return _transform_def(
        node_def,
        name=node_def.name,
        init_fn=init_fn,
        apply_fn=apply_fn,
        rebuild=lambda d: at(d, field=field),
    )
