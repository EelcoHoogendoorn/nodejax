from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _split_rng, _with_rng
from nodejax.generic import _over_generic
from nodejax.transforms.common import _mapped_init, _mapped_param_fn


@_over_generic
def ensemble(ndef: NodeDef, n: int | None = None) -> NodeDef:
    """vmap over the member axis: input broadcast, output stacked.

    A parametric node carries one param row per member — parameterize with
    stacked leaves, or declare the size n and parameterize(rng=key) to draw
    n independent members from the inner def's initializers. A
    NON-parametric cyclic node ensembles too: n independent state streams
    under the one broadcast input (declare n; there is no param axis to
    infer it from). A node with neither params nor state has no member
    identity to ensemble — use it directly.
    """
    if ndef.bound:
        raise TypeError('ensemble changes the meaning of param; apply it to the '
                        'NodeDef and parameterize with stacked params')
    if not (ndef.parametric or ndef.cyclic):
        raise TypeError(f'ensemble of {ndef!r}: no params and no state means no '
                        'member identity; use the node directly')
    if not ndef.parametric and n is None:
        raise TypeError(f'ensemble({ndef.name}) needs n=<member count>: with no '
                        'params, nothing else determines the ensemble size')

    def apply_fn(p, s, i):
        # apply-side rng: members must not share one draw — the boundary key
        # splits per member, each injected into that member's input
        p_axis = 0 if ndef.parametric else None
        if ndef.apply_takes_rng:
            count = jax.tree.leaves(p)[0].shape[0] if ndef.parametric else n
            keys, data = _split_rng(i, count)
            return jax.vmap(lambda p_, s_, k_: ndef.apply_fn(p_, s_, _with_rng(data, k_)),
                            in_axes=(p_axis, 0, 0))(p, s, keys)
        return jax.vmap(lambda p_, s_: ndef.apply_fn(p_, s_, i),
                        in_axes=(p_axis, 0))(p, s)

    if ndef.cyclic:
        init_fn = _mapped_init(ndef, n)
    else:
        def init_fn(nd_, p, state_input=Struct(), input=None):
            # the (empty) state carries no member axis; build it from one
            # member's params so a resolved def's init walk sees
            # member-shaped params, not stacked (mirrors stack)
            member0 = jax.tree.map(lambda x: x[0], p) if p != () else p
            return ndef.build_state(member0, state_input, input=input)

    return NodeDef(f'ensemble({ndef.name})', _mapped_param_fn(ndef, n),
                   init_fn,
                   apply_fn,
                   parametric=ndef.parametric, cyclic=ndef.cyclic,
                   apply_input_spec=ndef.apply_input_spec,
                   init_requires_input=ndef.init_requires_input,
                   init_reads_shape=ndef.init_reads_shape,
                   param_input_spec=ndef.param_input_spec if ndef.parametric else None,
                   state_input_spec=ndef.state_input_spec if ndef.cyclic else None)
