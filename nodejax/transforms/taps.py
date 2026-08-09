from __future__ import annotations

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, Composite, Serial, split_aux
from nodejax.generic import _over_generic
from nodejax.transforms.common import _over_bound


@_over_generic
@_over_bound
def taps(nd: NodeDef) -> NodeDef:
    """Observe every wire of a composite def: the output becomes
    (final carry, Struct(<each member's output, keyed by name>)) — the aux
    convention with every member opted in. Because taps are ordinary
    outputs, batch/ensemble/scan add their axes to them automatically.
    Shallow: this pipe's wires, not nested ones — tap an inner pipe
    before composing to see inside it."""
    if not isinstance(nd, Serial):
        raise TypeError(f'taps requires a serial pipe def (its members chain on '
                        f'the carry), got {nd!r}')
    members = nd.members
    names = list(members)

    def apply_fn(p, s, i):
        carry, states, aux = i, {}, {}
        for nm in names:
            states[nm], out = members[nm].apply_fn(p[nm], s[nm], carry)
            carry, member_aux = split_aux(out)
            aux[nm] = carry if member_aux is None else (carry, member_aux)
        return Struct(**states), (carry, Struct(**aux))

    out = NodeDef(f'taps({nd.name})', nd._param_impl, nd._init_impl, apply_fn,
                  nd.parametric, nd.cyclic, apply_input_spec=nd.apply_input_spec,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.init_reads_shape,
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None,
                  tags=nd.tags)
    return out
