from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.generic import _over_generic
from nodejax.transforms.common import _mapped_init, _mapped_param_fn


@_over_generic
def stack(ndef: NodeDef, n: int | None = None) -> NodeDef:
    """scan over the layer axis: layer k's output feeds layer k+1's input.

    param (and state) gain a leading layer axis; input/output shapes
    unchanged. parameterize with per-layer stacked leaves — or declare the
    depth n and parameterize(rng=key) to draw n layers from the inner def's
    declared initializers.
    """
    if ndef.bound:
        raise TypeError('stack changes the meaning of param; apply it to the '
                        'NodeDef and parameterize with per-layer stacked params')
    if not (ndef.parametric or ndef.cyclic):
        raise TypeError(f'stack requires a parametric or cyclic node, got {ndef!r}')

    def apply_fn(p, s, i):
        # apply-side rng: layers must not share one draw — the boundary key
        # splits per layer, each injected into that layer's carry
        if ndef.apply_takes_rng:
            keys, first = _split_rng(i, jax.tree.leaves(p)[0].shape[0])

            def step(carry, layer):
                p_, s_, k_ = layer
                s_new, out = ndef.apply_fn(p_, s_, _with_rng(carry, k_))
                return out, s_new
            out, new_states = jax.lax.scan(step, first, (p, s, keys))
            return new_states, out

        def step(carry, layer):
            p_, s_ = layer
            s_new, out = ndef.apply_fn(p_, s_, carry)
            return out, s_new
        out, new_states = jax.lax.scan(step, i, (p, s))
        return new_states, out

    if ndef.cyclic:
        init_fn = _mapped_init(ndef, n)
    else:
        def init_fn(nd_, p, state_input=Struct(), input=None):
            # a stateless stack's state carries no layer axis; build it
            # from one layer's params so spec propagation inside composite
            # inits sees per-layer shapes, not stacked
            layer0 = jax.tree.map(lambda x: x[0], p) if p != () else p
            return ndef.build_state(layer0, state_input,
                                    input=input)

    return NodeDef(f'stack({ndef.name})', _mapped_param_fn(ndef, n), init_fn, apply_fn,
                   ndef.parametric, ndef.cyclic, apply_input_spec=ndef.apply_input_spec,
                   init_requires_input=ndef.init_requires_input,
                  init_reads_shape=ndef.init_reads_shape,
                   param_input_spec=ndef.param_input_spec if ndef.parametric else None,
                   state_input_spec=ndef.state_input_spec if ndef.cyclic else None)
