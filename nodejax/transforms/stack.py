from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _split_rng, _with_rng, REQUIRED
from nodejax.generic import _over_generic
from nodejax.transforms.common import (_mapped_init, _mapped_param_fn,
                                       _scan_apply, _transform_def)


@_over_generic
def stack(node_def: NodeDef, n: int | None = None) -> NodeDef:
    """scan over the layer axis: layer k's output feeds layer k+1's input.

    param (and state) gain a leading layer axis; input/output shapes
    unchanged. parameterize with per-layer stacked leaves — or declare the
    depth n and parameterize(rng=key) to draw n layers from the inner def's
    declared initializers.
    """
    if node_def.bound:
        raise TypeError('stack changes the meaning of param; apply it to the '
                        'NodeDef and parameterize with per-layer stacked params')
    if not (node_def.parametric or node_def.cyclic):
        raise TypeError(f'stack requires a parametric or cyclic node, got {node_def!r}')

    def apply_fn(param, state, input):
        return _scan_apply(node_def, param, state, input, stacked_param=True)

    return _transform_def(
        node_def,
        name=f'stack({node_def.name})',
        param_fn=_mapped_param_fn(node_def, n),
        init_fn=_mapped_init(node_def, n),
        apply_fn=apply_fn,
    )
