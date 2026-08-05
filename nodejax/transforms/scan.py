from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import Node, NodeDef, _trivial_init_fn, _input_or_none, _resolve, _has_rng
from nodejax.generic import _over_generic
from nodejax.spec import materialize
from nodejax.transforms.common import _split, _rewrap


@_over_generic
def scan(node: NodeDef | Node, record: bool = False,
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
    nd, param = _split(node)
    if not nd.cyclic:
        raise TypeError(f'scan requires a cyclic node, got {nd!r}')

    # scan's input contract is a stream, the inner node's is a step: the
    # inner def resolves against ONE element of the bound stream
    def param_fn(ndef, param_input=Struct()):
        stream = _input_or_none(ndef)
        d = (nd if stream is None
             else _resolve(nd, jax.tree.map(lambda x: x[0], stream)))
        return d.build_param(param_input)

    def _divert_rng(inputs):
        if _has_rng(inputs):
            return (Struct(rng=inputs['rng']),
                    inputs.without('rng'))
        return Struct(), inputs

    def _loop(p, s0, inputs):
        if record:
            def step(s_, i_):
                s2, out = nd.apply_fn(p, s_, i_)
                return s2, Struct(state=s2, output=out)
            return jax.lax.scan(step, s0, inputs)
        return jax.lax.scan(lambda s_, i_: nd.apply_fn(p, s_, i_), s0, inputs)

    if persist is None:
        def apply_fn(p, s, inputs):
            seed, inputs = _divert_rng(inputs)
            # the first sequence element seeds the internalized init, so
            # input-shaped state (previous-error registers, feedback seeds)
            # derives from the sequence itself — no shape statics
            element = jax.tree.map(lambda x: x[0], inputs)
            s0 = _resolve(nd, element).build_state(p, seed, input=element)
            _, ys = _loop(p, s0, inputs)
            return (), ys

        out = NodeDef(f'scan({nd.name})', param_fn, _trivial_init_fn, apply_fn,
                      nd.parametric, cyclic=False,
                      param_input_spec=nd.param_input_spec if nd.parametric else None)
        return _rewrap(out, param)

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
            return nd.build_state(p, state_input)
        element = jax.tree.map(lambda x: x[0], seq)
        return _resolve(nd, element).build_state(p, state_input, input=element)

    def apply_fn(p, s_outer, inputs):
        seed, inputs = _divert_rng(inputs)
        element = jax.tree.map(lambda x: x[0], inputs)
        fresh = _resolve(nd, element).build_state(p, seed, input=element)
        final, ys = _loop(p, merge(fresh, s_outer), inputs)
        return final, ys

    out = NodeDef(f'scan({nd.name})', param_fn, init_fn, apply_fn,
                  nd.parametric, cyclic=True,
                  init_requires_input=nd.init_requires_input,
                  init_reads_shape=nd.init_reads_shape,
                  param_input_spec=nd.param_input_spec if nd.parametric else None,
                  state_input_spec=nd.state_input_spec if nd.cyclic else None)
    return _rewrap(out, param)
