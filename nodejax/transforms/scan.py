from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _trivial_init_fn, _input_or_none, _resolve, _has_rng
from nodejax.generic import _over_generic
from nodejax.spec import materialize
from nodejax.transforms.common import _over_bound, _transform_def


@_over_generic
@_over_bound
def scan(node_def: NodeDef, record: bool = False,
         persist: tuple[str, ...] | None = None) -> NodeDef | Node:
    """Internalize the state loop: a step-level cyclic node becomes a
    sequence-level non-cyclic one (CN -> N, PCN -> PN in lattice terms).

    State starts from init defaults and stays inside. With record=True the
    per-step output becomes Struct(state=..., output=...) — the full state
    trajectory rides along for plotting/analysis, doctrine-compliant
    (a Struct, not a tuple).

    A reserved 'rng' field of the input Struct — a single key, no time
    axis (entropy cannot ride the xs) — is diverted to the internalized
    init, so stochastic state under a scanned rollout initializes from
    the same one-shot rng-as-input entry point as everything else.

    persist splits FAST state from SLOW state. It is a MAPPING from
    write slot to read slot, applied at episode start by exact path
    segment ('norm' does not match 'denormalizer'):

        persist={'stats': 'stats'}                  # slot carries itself
        persist={'frozen': 'stats', 'stats': 'stats'}
            # 'frozen' REFRESHES from the carried 'stats' sibling —
            # within-episode reads never see within-episode updates
            # (true frozen-read evaluation under a scan; the same shape
            # as a target network refreshing from the live one)
        # unmatched slots re-initialize fresh (recurrent activations,
        # episode registers); a bare tuple of names is identity sugar

    The result stays CYCLIC, with the full step state as its state —
    internalizing everything is a choice, and for slow state the wrong
    one; the enclosing loop (a trainer, an outer scan) carries the slow
    state to the next episode. Fresh slots ride along structurally but
    their persisted values are ignored (each apply overwrites them from
    init, so batch-shaped fast state tolerates shape changes between
    applies). A callable merge(fresh, outer) -> episode-start state
    remains the general escape hatch."""
    if not node_def.cyclic:
        raise TypeError(f'scan requires a cyclic node, got {node_def!r}')

    # scan's input contract is a stream, the inner node's is a step: the
    # inner def resolves against ONE element of the bound stream
    def param_fn(ndef, param_input=Struct()):
        stream = _input_or_none(ndef)
        d = (node_def if stream is None
             else _resolve(node_def, jax.tree.map(lambda x: x[0], stream)))
        return d.build_param(param_input)

    def _divert_rng(inputs):
        if _has_rng(inputs):
            return (Struct(rng=inputs['rng']),
                    inputs.without('rng'))
        return Struct(), inputs

    def _loop(p, s0, inputs):
        if record:
            def step(s_, i_):
                s2, out = node_def.apply_fn(p, s_, i_)
                return s2, Struct(state=s2, output=out)
            return jax.lax.scan(step, s0, inputs)
        return jax.lax.scan(lambda s_, i_: node_def.apply_fn(p, s_, i_), s0, inputs)

    if persist is None:
        def apply_fn(nd, p, s, inputs):
            seed, inputs = _divert_rng(inputs)
            # the first sequence element seeds the internalized init, so
            # input-shaped state (previous-error registers, feedback seeds)
            # derives from the sequence itself — no shape statics
            element = jax.tree.map(lambda x: x[0], inputs)
            s0 = _resolve(node_def, element).build_state(p, seed, input=element)
            _, ys = _loop(p, s0, inputs)
            return (), ys

        return _transform_def(
            node_def,
            name=f'scan({node_def.name})',
            param_fn=param_fn,
            init_fn=_trivial_init_fn,
            apply_fn=apply_fn,
            cyclic=False,
            apply_input_spec=None,
        )

    if callable(persist):
        merge = persist
    else:
        mapping = dict(persist) if isinstance(persist, dict) else {t: t for t in persist}

        def _segments(key_path):
            return tuple(getattr(k, 'name', getattr(k, 'key', None)) for k in key_path)

        def merge(fresh, outer):
            """The episode-start state, decided leaf by leaf: a leaf whose
            path crosses a slot named in the mapping is READ from the
            carried `outer` state — at the same path with that slot renamed
            to its mapped source, so 'frozen' <- 'stats' reads the sibling
            slot — and every other leaf keeps its fresh re-initialized
            value. Paths compare by NAME segments, so the rule is
            structural (a slot name anywhere in the tree), not positional."""
            fresh_leaves, treedef = jax.tree_util.tree_flatten_with_path(fresh)
            carried = {_segments(path): leaf
                       for path, leaf in jax.tree_util.tree_flatten_with_path(outer)[0]}
            leaves = []
            for path, fresh_leaf in fresh_leaves:
                segments = _segments(path)
                hit = next(((i, s) for i, s in enumerate(segments) if s in mapping), None)
                if hit is None:
                    leaves.append(fresh_leaf)                # unmapped: episode restart
                    continue
                i, slot = hit
                source = segments[:i] + (mapping[slot],) + segments[i + 1:]
                if source not in carried:
                    raise TypeError(f"persist mapping '{slot} <- {mapping[slot]}' "
                                    f"has no source slot at {source}")
                leaves.append(carried[source])               # mapped: carried across
            return jax.tree_util.tree_unflatten(treedef, leaves)

    def init_fn(ndef, p, state_input=Struct(), input=None):
        seq = input if input is not None else _input_or_none(ndef)
        if seq is None:
            return node_def.build_state(p, state_input)
        element = jax.tree.map(lambda x: x[0], seq)
        return _resolve(node_def, element).build_state(p, state_input, input=element)

    def apply_fn(nd, p, s_outer, inputs):
        seed, inputs = _divert_rng(inputs)
        element = jax.tree.map(lambda x: x[0], inputs)
        fresh = _resolve(node_def, element).build_state(p, seed, input=element)
        final, ys = _loop(p, merge(fresh, s_outer), inputs)
        return final, ys

    return _transform_def(
        node_def,
        name=f'scan({node_def.name})',
        param_fn=param_fn,
        init_fn=init_fn,
        apply_fn=apply_fn,
        cyclic=True,
        apply_input_spec=None,
    )
