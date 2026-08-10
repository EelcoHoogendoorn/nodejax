from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.generic import _over_generic
from nodejax.transforms.common import _mapped_init, _mapped_param_fn, _transform_def, _vmap_apply


@_over_generic
def ensemble(node_def: NodeDef, n: int | None = None,
             axis: str = 'ensemble') -> NodeDef:
    """vmap over the member axis: input broadcast, output stacked.

    A parametric node carries one param row per member — parameterize with
    stacked leaves, or declare the size n and parameterize(rng=key) to draw
    n independent members from the inner def's initializers. A
    NON-parametric cyclic node ensembles too: n independent state streams
    under the one broadcast input (declare n; there is no param axis to
    infer it from). A node with neither params nor state has no member
    identity to ensemble — use it directly.

    The member axis is NAMED (default 'ensemble', the reserved
    convention), so members reduce across the population with jax
    collectives; a declared axis need for the name is satisfied here,
    and re-binding a name already bound inside refuses.
    """
    if node_def.bound:
        raise TypeError('ensemble changes the meaning of param; apply it to the '
                        'NodeDef and parameterize with stacked params')
    if not (node_def.parametric or node_def.cyclic):
        raise TypeError(f'ensemble of {node_def!r}: no params and no state means no '
                        'member identity; use the node directly')
    if not node_def.parametric and n is None:
        raise TypeError(f'ensemble({node_def.name}) needs n=<member count>: with no '
                        'params, nothing else determines the ensemble size')

    num_members = n
    def apply_fn(param, state, input):
        param_axis = 0 if node_def.parametric else None
        count = jax.tree.leaves(param)[0].shape[0] if node_def.parametric else num_members
        return _vmap_apply(
            node_def, param, state, input,
            param_axis=param_axis, state_axis=0, input_axis=None,
            axis_name=axis, count=count,
        )

    return _transform_def(
        node_def,
        name=f'ensemble({node_def.name})',
        param_fn=_mapped_param_fn(node_def, n),
        init_fn=_mapped_init(node_def, num_members),
        apply_fn=apply_fn,
    )
