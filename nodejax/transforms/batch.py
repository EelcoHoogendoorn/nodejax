from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core import (Node, NodeDef, _input_or_none, _resolve,
                                _split_rng, _with_rng, _spec_resolved)
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.transforms.common import _over_bound, _tile_state, _transform_def, _mapped_apply_fn


from nodejax.transforms.tree_utils import map_node_leaves, map_state_leaves


@_over_generic
@_over_bound
def batch(node_def: NodeDef, n: int | None = None,
          axis: str = 'batch') -> NodeDef:
    """vmap over the input axis: params broadcast, input/output/state batched.

    Type-preserving; accepts defs and bound nodes (param meaning unchanged).
    For cyclic nodes the per-element state is tiled to the batch size: read
    from the bound batched input shape, or from the static n=<batch size>
    here when the node inits without a shape (a shape-free, constant state).

    The vmap axis is NAMED (default 'batch', the reserved convention), so
    members reduce across it with jax collectives. Nodes tagged with
    'single_batch_state' retain a single unbatched state across the batch.
    """
    state_in = map_node_leaves(node_def, lambda member: None if 'single_batch_state' in member.tags else 0)

    def apply_fn(nd, param, state, input):
        # state_in is both in_axes and out_axes: a 'single_batch_state' member
        # is broadcast in and comes back unmapped, no slicing after the fact
        return _mapped_apply_fn(
            node_def, param, state, input,
            param_axis=None, state_axis=state_in, input_axis=0,
            axis_name=axis,
        )

    if node_def.cyclic:
        def init_fn(ndef, param, state_input=Struct(), input=None):
            # per-element shape and batch size are STATIC: the shape from the
            # bound (batched) def, the count from its leading axis (or the
            # construction n= for a shape-free node) — never a runtime value
            batched = input if input is not None else _input_or_none(ndef)
            if batched is not None:
                rows = jax.tree.leaves(batched)[0].shape[0]
                if n is not None and n != rows:
                    raise TypeError(f'batch({node_def.name}): n={n} conflicts with the bound '
                                    f'batched axis {rows}')
                count = rows
                element = jax.tree.map(lambda leaf: leaf[0], batched)
                seed_state = _resolve(node_def, element).build_state(param, state_input, input=element)
            elif n is not None:
                count = n
                seed_state = node_def.build_state(param, state_input)
            else:
                raise TypeError(f'batch({node_def.name}).init needs a bound batched input '
                                'shape (with_input(<batched spec>)), or batch(node, '
                                'n=<batch size>) for a shape-free node')
            return map_state_leaves(
                node_def, seed_state,
                lambda member, m_state: m_state if 'single_batch_state' in member.tags else _tile_state(m_state, count))
    else:
        init_fn = node_def._init_impl

    if node_def.parametric:
        def param_fn(ndef, param_input):
            batched = _input_or_none(ndef)
            if batched is not None:
                inner_spec = _input_or_none(node_def)
                leaf_b = jax.tree.leaves(batched)[0]
                if inner_spec is not None and leaf_b.ndim == jax.tree.leaves(inner_spec)[0].ndim:
                    element = batched
                elif leaf_b.ndim > 1:
                    element = jax.tree.map(lambda leaf: leaf[0], batched)
                else:
                    element = batched
                return _resolve(node_def, element).build_param(param_input)
            return node_def.build_param(param_input)
    else:
        param_fn = node_def._param_impl

    return _transform_def(
        node_def,
        name=f'batch({node_def.name})',
        param_fn=param_fn,
        init_fn=init_fn,
        apply_fn=apply_fn,
        apply_input_spec=node_def.apply_input_spec if not _spec_resolved(node_def.apply_input_spec) else None,
        init_reads_shape=node_def.cyclic,
        rebuild=lambda d: batch(d, n=n, axis=axis),
    )

