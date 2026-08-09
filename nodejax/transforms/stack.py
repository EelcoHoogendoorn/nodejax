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

    def apply_fn(param, state, input):
        from nodejax.core import split_aux
        # apply-side rng: layers must not share one draw — the boundary key
        # splits per layer, each injected into that layer's carry
        if ndef.apply_takes_rng:
            keys, first = _split_rng(input, jax.tree.leaves(param)[0].shape[0])

            def step(carry, layer):
                layer_param, layer_state, layer_key = layer
                new_state, out = ndef.apply_fn(layer_param, layer_state, _with_rng(carry, layer_key))
                clean_out, aux = split_aux(out)
                return clean_out, (new_state, aux)
            out, (new_states, auxs) = jax.lax.scan(step, first, (param, state, keys))
            return new_states, (out, auxs) if auxs is not None else out

        def step(carry, layer):
            layer_param, layer_state = layer
            new_state, out = ndef.apply_fn(layer_param, layer_state, carry)
            clean_out, aux = split_aux(out)
            return clean_out, (new_state, aux)
        out, (new_states, auxs) = jax.lax.scan(step, input, (param, state))
        return new_states, (out, auxs) if auxs is not None else out

    if ndef.cyclic:
        init_fn = _mapped_init(ndef, n)
    else:
        def init_fn(def_obj, param, state_input=Struct(), input=None):
            # a stateless stack's state carries no layer axis; build it
            # from one layer's params so spec propagation inside composite
            # inits sees per-layer shapes, not stacked
            layer0 = jax.tree.map(lambda leaf: leaf[0], param) if param != () else param
            return ndef.build_state(layer0, state_input,
                                    input=input)

    return NodeDef(f'stack({ndef.name})', _mapped_param_fn(ndef, n), init_fn, apply_fn,
                   ndef.parametric, ndef.cyclic, apply_input_spec=ndef.apply_input_spec,
                   init_requires_input=ndef.init_requires_input,
                   init_reads_shape=ndef.init_reads_shape,
                   param_input_spec=ndef.param_input_spec if ndef.parametric else None,
                   state_input_spec=ndef.state_input_spec if ndef.cyclic else None,
                   tags=ndef.tags)
