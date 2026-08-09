from __future__ import annotations

import jax

from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.transforms.common import _over_bound, _mapped_init


@_over_generic
@_over_bound
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

    def apply_fn(param, state, input):
        # apply-side rng: positions must not share one draw — the boundary
        # key splits per position, each injected into that position's carry
        if node_def.apply_takes_rng:
            keys, first = _split_rng(input, count)

            def step(carry, xs):
                pos_state, pos_key = xs
                new_state, out = node_def.apply_fn(param, pos_state, _with_rng(carry, pos_key))
                return out, new_state
            out, new_states = jax.lax.scan(step, first, (state, keys), length=count)
            return new_states, out

        def step(carry, pos_state):
            new_state, out = node_def.apply_fn(param, pos_state, carry)
            return out, new_state
        out, new_states = jax.lax.scan(step, input, state, length=count)
        return new_states, out

    # one param, n state slots: the shared-params branch of the mapped init
    init_fn = _mapped_init(node_def, count, stacked=False) if node_def.cyclic else node_def._init_impl

    out = NodeDef(f'repeat({node_def.name})', node_def._param_impl, init_fn, apply_fn,
                  node_def.parametric, node_def.cyclic, apply_input_spec=node_def.apply_input_spec,
                  init_requires_input=node_def.init_requires_input,
                  init_reads_shape=node_def.init_reads_shape,
                  param_input_spec=node_def.param_input_spec if node_def.parametric else None,
                  state_input_spec=node_def.state_input_spec if node_def.cyclic else None,
                  tags=node_def.tags)
    return out
