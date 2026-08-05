from __future__ import annotations

import jax

from nodejax.core import (Node, NodeDef, _input_or_none, _resolve,
                                _split_rng, _with_rng, _spec_resolved)
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.transforms.common import _split, _rewrap, _tile_state


@_over_generic
def batch(node: NodeDef | Node, n: int | None = None) -> NodeDef | Node:
    """vmap over the input axis: params broadcast, input/output/state batched.

    Type-preserving; accepts defs and bound nodes (param meaning unchanged).
    For cyclic nodes the per-element state is tiled to the batch size: read
    from the bound batched input shape, or from the static n=<batch size>
    here when the node inits without a shape (a shape-free, constant state).
    """
    nd, param = _split(node)

    def apply_fn(p, s, i):
        # apply-side rng: elements must not share one draw — the boundary key
        # splits per element, each injected into that element's input slice
        if nd.apply_takes_rng:
            data = i.without('rng')
            keys = jax.random.split(i.rng, jax.tree.leaves(data)[0].shape[0])
            return jax.vmap(lambda s_, d_, k_: nd.apply_fn(
                p, s_, _with_rng(d_, k_)))(s, data, keys)
        return jax.vmap(lambda s_, i_: nd.apply_fn(p, s_, i_))(s, i)

    if nd.cyclic:
        def init_fn(ndef, p, state_input=Struct(), input=None):
            # per-element shape and batch size are STATIC: the shape from the
            # bound (batched) def, the count from its leading axis (or the
            # construction n= for a shape-free node) — never a runtime value
            batched = input if input is not None else _input_or_none(ndef)
            if batched is not None:
                axis = jax.tree.leaves(batched)[0].shape[0]
                if n is not None and n != axis:
                    raise TypeError(f'batch({nd.name}): n={n} conflicts with the bound '
                                    f'batched axis {axis}')
                count = axis
                element = jax.tree.map(lambda x: x[0], batched)
                s0 = _resolve(nd, element).build_state(p, state_input, input=element)
            elif n is not None:
                count = n
                s0 = nd.build_state(p, state_input)
            else:
                raise TypeError(f'batch({nd.name}).init needs a bound batched input '
                                'shape (with_input(<batched spec>)), or batch(node, '
                                'n=<batch size>) for a shape-free node')
            return _tile_state(s0, count)
    else:
        init_fn = nd._init_impl

    out = NodeDef(f'batch({nd.name})', nd._param_impl, init_fn, apply_fn, nd.parametric, nd.cyclic,
                  # the inner's UNRESOLVED field spec passes through (field
                  # identity, incl. rng, holds under batching); a resolved
                  # per-element shape would lie about the batched axis
                  apply_input_spec=nd.apply_input_spec
                  if not _spec_resolved(nd.apply_input_spec) else None,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.cyclic,   # its init sizes from the batched spec
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None)
    return _rewrap(out, param)
