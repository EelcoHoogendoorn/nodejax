from __future__ import annotations

import jax

from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.transforms.common import _split, _rewrap, _mapped_init


@_over_generic
def repeat(node: NodeDef | Node, n: int) -> NodeDef | Node:
    """Apply the SAME node n times in sequence: weight-tied depth.

    Where stack gives every layer its own params (a leading layer axis),
    repeat threads one set of params through n applications — iterated
    refinement. Param meaning is unchanged, so bound Nodes rebind. Cyclic
    nodes keep one state slot PER position (state gains a leading axis of
    n): tied weights, untied state. Output must be input-shaped — the same
    carry contract as stack.
    """
    nd, param = _split(node)

    def apply_fn(p, s, i):
        # apply-side rng: positions must not share one draw — the boundary
        # key splits per position, each injected into that position's carry
        if nd.apply_takes_rng:
            keys, first = _split_rng(i, n)

            def step(carry, xs):
                s_, k_ = xs
                s_new, out = nd.apply_fn(p, s_, _with_rng(carry, k_))
                return out, s_new
            out, new_states = jax.lax.scan(step, first, (s, keys), length=n)
            return new_states, out

        def step(carry, s_):
            s_new, out = nd.apply_fn(p, s_, carry)
            return out, s_new
        out, new_states = jax.lax.scan(step, i, s, length=n)
        return new_states, out

    # one param, n state slots: the shared-params branch of the mapped init
    init_fn = _mapped_init(nd, n, stacked=False) if nd.cyclic else nd._init_impl

    out = NodeDef(f'repeat({nd.name})', nd._param_impl, init_fn, apply_fn,
                  nd.parametric, nd.cyclic, apply_input_spec=nd.apply_input_spec,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.init_reads_shape,
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None)
    return _rewrap(out, param)
