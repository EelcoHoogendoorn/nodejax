from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, Composite, _input_or_none
from nodejax.authoring import KeyStream
from nodejax.spec import materialize


def parallel(**named: GenericDef | NodeDef | Node) -> GenericDef | NodeDef | Node:
    """Compose named nodes over the strands of a Struct signal — the
    product twin of serial: where a pipe chains members over one
    signal, parallel(**members) runs each member on the input field of
    its own name and emits the outputs under the same names. Strict by
    construction: the input's fields and the member names must match
    exactly, so every strand is written down — a passthrough strand is
    an explicit identity member, never an omission.

    Param and state compose as Structs keyed by strand name, exactly
    as pipes key by member name; all defs -> NodeDef, all bound ->
    bound Node. Strands are independent: init offers each member its
    own field of the offered input, and rng splits route per member."""
    if not named:
        raise TypeError('parallel() needs at least one member')
    from nodejax.generic import GenericDef
    from nodejax.compose import _as_generic
    if any(isinstance(v, GenericDef) for v in named.values()):
        generic_members = {nm: _as_generic(v) for nm, v in named.items()}

        def specialize_fn(**statics: Any) -> NodeDef | Node:
            unknown = set(statics) - set(generic_members)
            if unknown:
                raise TypeError(f'unknown parallel generic members: {sorted(unknown)}')
            specialized = {nm: generic_members[nm].specialize(**statics.get(nm, {}))
                           for nm in generic_members}
            return parallel(**specialized)

        return GenericDef('(' + ' | '.join(named) + ')', specialize_fn, members=generic_members)

    if not all(v.bound for v in named.values()):
        promoted = {}
        for nm, v in named.items():
            if v.bound:
                if v.param != ():
                    raise TypeError(f"cannot mix bound parametric node '{nm}' into an "
                                    'unbound parallel block; compose defs and '
                                    'parameterize the block instead')
                v = v.ndef
            promoted[nm] = v
        return _Parallel(promoted)
    if all(v.bound for v in named.values()):
        defs = {nm: v.ndef for nm, v in named.items()}
        return _Parallel(defs).bind(Struct(**{nm: v.param for nm, v in named.items()}))
    raise TypeError('parallel() members must be Nodes, NodeDefs, or GenericDefs')


def _Parallel(defs: dict[str, NodeDef]) -> NodeDef:
    names = list(defs)

    def param_fn(ndef, param_input=Struct()):
        key = param_input.rng if 'rng' in param_input else None
        parts = {}
        for nm in names:
            d = defs[nm]
            if not d.parametric:
                parts[nm] = ()
                continue
            fields = (dict(param_input[nm].__items__)
                      if nm in param_input else {})
            if key is not None and 'rng' in d.param_input_spec and 'rng' not in fields:
                key, fields['rng'] = jax.random.split(key)
            try:
                parts[nm] = d.build_param(Struct(**fields))
            except TypeError as e:
                raise TypeError(f"parallel member '{nm}': {e}") from e
        return Struct(**parts)

    def member_param(param, nm):
        # the trivial param of a nonparametric block is (), not a Struct
        return () if param == () else param[nm]

    def init_fn(ndef, param, state_input=Struct(), input=None):
        src = input if input is not None else _input_or_none(ndef)
        carry = materialize(src) if src is not None else None
        key = state_input.rng if 'rng' in state_input else None
        states = {}
        for nm in names:
            d = defs[nm]
            seed = state_input[nm] if nm in state_input else Struct()
            fields = dict(seed.__items__)
            if key is not None and 'rng' in d.state_input_spec and 'rng' not in fields:
                key, fields['rng'] = jax.random.split(key)
            strand_in = carry[nm] if carry is not None and d.cyclic else None
            try:
                states[nm] = d.build_state(member_param(param, nm), Struct(**fields),
                                           input=strand_in)
            except TypeError as e:
                raise TypeError(f"parallel member '{nm}': {e}") from e
        return Struct(**states)

    # apply-side rng bubbles: the block consumes rng iff a strand's apply
    # does — one boundary key in the input bundle, split per consuming strand
    rng_strands = {nm: defs[nm].apply_takes_rng for nm in names}
    boundary_rng = any(rng_strands.values())

    def apply_fn(nd, param, state, input):
        key = None
        if boundary_rng:
            key = input.rng                  # missing key fails here, loudly
            input = input.without('rng')
        extra = set(input.__keys__) - set(names)
        if extra:
            raise TypeError(f'parallel block has no member for input fields {sorted(extra)}')
        states, outs = {}, {}
        for nm in names:
            strand_in = input[nm]
            if rng_strands[nm]:
                key, sub = jax.random.split(key)
                if not isinstance(strand_in, Struct):
                    raise TypeError(f"parallel strand '{nm}' consumes rng from its "
                                    'input; its field must be a named Struct to '
                                    'carry the key alongside the data')
                strand_in = Struct(**dict(strand_in.__items__), rng=sub)
            states[nm], outs[nm] = defs[nm].apply_fn(member_param(param, nm), state[nm], strand_in)
        return Struct(**states), Struct(**outs)

    return Composite('(' + ' | '.join(names) + ')', param_fn, init_fn, apply_fn,
                     parametric=any(d.parametric for d in defs.values()),
                     cyclic=any(d.cyclic for d in defs.values()),
                     members=dict(defs),
                     rebuild=lambda new: _Parallel(dict(new)))
