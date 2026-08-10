"""Composition: serial pipes and hand-wired composites over named members.

Param and state compose as Structs keyed by member name — including trivial
()s, so no member kind is special. A pipe of generics is itself generic,
its statics the nested tree of member statics. `>>` flattens across
nesting and dispatches to the right stage.
"""

from __future__ import annotations

import inspect
import re
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct, Aux
from nodejax.core import (Node, NodeDef, Composite, Serial, split_aux, _input_or_none,
                                _resolve, _bundle_spec_from_sig, REQUIRED,
                                _hoist_apply_rng, hoist_rng, _has_rng)
from nodejax.authoring import (KeyStream, _lift_param, _lift_init,
                                     _state_spec_from_sig, _init_requires_input)
from nodejax.generic import GenericDef
from nodejax.wiring import (_author_call, _wrap_apply, _Wired, _InitWired, _BuildingWired,
                                  _Solo, _NO_INPUT)
from nodejax.spec import materialize


def serial(**named: NodeDef | Node) -> NodeDef | Node:
    """Compose named nodes into a serial pipe.

    All members Node -> bound Node; otherwise a NodeDef. Param and
    state compose as Structs keyed by member name — including trivial
    ()s, so no member kind is special. Bound parametric members are
    transport containers: the def joins the pipe, the params become
    stored construction values that parameterize uses wherever kwargs
    leave the slot open — a >> b()(gain=2.0) spells the member tree
    once, at the composition site."""
    if not named:
        raise TypeError('serial() needs at least one member')
    if all(v.bound for v in named.values()):
        defs = {nm: v.ndef for nm, v in named.items()}
        return _serial(defs).bind(Struct(**{nm: v.param for nm, v in named.items()}))
    defs, given = _promote_members(named)
    return _serial(defs, given)


def serial_generic(**named: GenericDef) -> GenericDef:
    """Compose named generics into a serial pipe that is itself generic:
    specialize takes nested statics keyed by member name (members needing
    no statics may be omitted), and returns the specialized serial pipe."""
    names = list(named)

    def specialize_fn(**statics: Any) -> NodeDef | Node:
        unknown = set(statics) - set(names)
        if unknown:
            raise TypeError(f'unknown generic pipe members: {sorted(unknown)}')
        return serial(**{nm: named[nm].specialize(**statics.get(nm, {})) for nm in names})

    return GenericDef('(' + ' >> '.join(names) + ')', specialize_fn, members=dict(named))


def _as_generic(x: GenericDef | NodeDef | Node) -> GenericDef:
    """Promote a pipe member to the generic stage: already-specialized
    members become constant generics taking no statics."""
    if isinstance(x, GenericDef):
        return x
    nd = x.ndef
    if x.bound and x.param != ():
        raise TypeError(f"cannot lift bound parametric node '{nd.name}' into a generic pipe")

    def constant(**statics: Any) -> NodeDef:
        # broadcasts ('*.name') are to-whom-it-may-concern; constants are
        # not concerned — explicit statics remain an error
        statics = {k: v for k, v in statics.items() if not k.startswith('*.')}
        if statics:
            raise TypeError(f"pipe member '{nd.name}' takes no static arguments")
        return nd

    return GenericDef(nd.name, constant)


def _given_defaults(spec: Struct, given: dict[str, Any]) -> Struct:
    """Overlay stored constructions onto a composite's bundle spec: each
    pre-bound member's slot carries its finished param as the slot DEFAULT
    (whole-slot replacement — a finished param is not a field spec to merge
    into). The spec thereby states what binding made optional."""
    merged = dict(spec.__items__)
    merged.update(given)
    return Struct(**merged)


def _probe_seed(d: NodeDef) -> Struct:
    """The seed for a state built only to walk PAST a member: a throwaway
    key where its bundle wants one, nothing else. The probe state is
    discarded (the param walk keeps only the carry it advances)."""
    return (Struct(rng=jax.random.PRNGKey(0))
            if 'rng' in d.state_input_spec else Struct())


import re


def _probe_apply(apply_fn: Callable, param: Any, state: Any, input: Any) -> Any:
    """Run apply_fn during shape/state probing walks, dynamically binding size-1
    dummy vmap axes if JAX raises `NameError: unbound axis name: <name>`."""
    bound_axes = set()
    fn = apply_fn
    while True:
        try:
            return fn(param, state, input)
        except NameError as e:
            match = re.search(r"unbound axis name:\s*(\w+)", str(e))
            if match and match.group(1) not in bound_axes:
                axis_name = match.group(1)
                bound_axes.add(axis_name)
                prev_fn = fn
                def make_bound(inner_fn, name):
                    def bound_fn(p_, s_, i_):
                        lead = lambda t: jax.tree.map(lambda x: jnp.asarray(x)[None], t)
                        out = jax.vmap(inner_fn, axis_name=name)(lead(p_), lead(s_), lead(i_))
                        return jax.tree.map(lambda x: x[0], out)
                    return bound_fn
                fn = make_bound(prev_fn, axis_name)
            else:
                raise e


def _step_carry(d: NodeDef, param: Any, state: Any, carry: Any, nm: str) -> Any:
    """Advance the walk's carry through one member: run its apply on the
    carry and pass the clean output on (aux diverted off). A member that
    consumes apply-rng gets a probe key (the walk derives shapes; the draw
    is discarded). Failures name the member — a shape mismatch surfaces at the
    member that broke, not downstream."""
    if d.apply_takes_rng and not _has_rng(carry):
        carry = carry.replace(rng=jax.random.PRNGKey(0))   # probe key; discarded
    try:
        _, out = _probe_apply(d.apply_fn, param, state, carry)
    except Exception as e:
        raise TypeError(f"walk failed at member '{nm}': {e}") from e
    return split_aux(out)[0]


def _member_param(nm: str, d: NodeDef, sub_bundle: Any, key: Any, input: Any = None):
    """Construct one member's params through the entry: the sub-bundle
    (a Struct, or None for a slot the bundle leaves open) becomes the member's
    bundle, extended with a split of the composite key when the member's
    bundle spec wants rng. A call-site carry resolves the member's def, and a
    shape-reading ctor reads it there — nothing reserved rides the bundle.
    Returns (param, key), the key advanced iff a split was routed."""
    if not d.parametric:
        return (), key
    bundle = sub_bundle if sub_bundle is not None else Struct()
    if key is not None and (d.param_input_spec is None or _has_rng(d.param_input_spec)) and 'rng' not in bundle:
        key, sub_ = jax.random.split(key)
        bundle = bundle.replace(rng=sub_)
    if input is not None:
        d = _resolve(d, input)              # the member reads its call-site shape via its def
    try:
        return d.build_param(bundle), key
    except TypeError as e:
        raise TypeError(f"member '{nm}': {e}") from e


def _serial(defs: dict[str, NodeDef], given: dict[str, Any] | None = None) -> NodeDef:
    """Build the def of a serial pipe over named member defs;
    `given` carries stored construction values per member."""
    names = list(defs)
    given = given or {}

    def param_fn(ndef, param_input=Struct()):
        """The pipe's param constructor: member slots of the bundle are
        member param inputs, a boundary rng field splits toward members whose
        bundles want one, and — when the pipe's def carries an input spec — the
        carry is threaded member by member (each member's def resolved to its
        own upstream shape, each probed init+apply deriving the next member's).
        Stored constructions (bound members at composition) fill slots the
        bundle leaves open."""
        key = param_input.rng if 'rng' in param_input else None
        carry = _input_or_none(ndef)
        parts = {}
        for nm in names:
            d = defs[nm]
            sub_bundle = param_input[nm] if nm in param_input else None
            if sub_bundle is None and nm in given:
                parts[nm] = given[nm]         # stored construction values
            else:
                parts[nm], key = _member_param(nm, d, sub_bundle, key, input=carry)
            if carry is not None:             # the shape walk: derive the next shape
                s = _resolve(d, carry).build_state(parts[nm], _probe_seed(d), input=carry)
                carry = _step_carry(d, parts[nm], s, carry, nm)
        return Struct(**parts)

    def init_fn(ndef, param, state_input=Struct(), input=None):
        """The pipe's init: member slots of the seed bundle are the
        members' seeds, a boundary rng field splits toward members whose
        bundles want one, and the input value (or the def's bound spec)
        threads member by member — each member's def resolved to its own
        upstream shape, each apply deriving the next member's carry."""
        carry = input if input is not None else _input_or_none(ndef)
        key = state_input.rng if 'rng' in state_input else None
        states = {}
        for nm in names:
            d = defs[nm]
            seed = state_input[nm] if nm in state_input else Struct()
            if key is not None and 'rng' in d.state_input_spec and 'rng' not in seed:
                key, sub_ = jax.random.split(key)
                seed = seed.replace(rng=sub_)
            d2 = d if carry is None else _resolve(d, carry)
            try:
                states[nm] = d2.build_state(param[nm], seed, input=carry)
            except TypeError as e:
                raise TypeError(f"pipe member '{nm}': {e}") from e
            if carry is not None:
                carry = _step_carry(d, param[nm], states[nm], carry, nm)
        return Struct(**states)

    # apply-side rng requirement bubbles: the pipe consumes rng iff a member's
    # apply does. The caller passes ONE key in its input bundle; the pipe
    # splits it toward each consuming member, injected as that member's rng
    # input field (entropy never rides the wire — an upstream member does not
    # emit keys).
    rng_members = {nm: defs[nm].apply_takes_rng for nm in names}
    boundary_rng = any(rng_members.values())

    def apply_fn(nd, param, state, input):
        """Members chain on the carry; a member returning (output, aux)
        has the aux diverted into the pipe's own collection, keyed by
        member name, while the clean signal flows on. If any aux was
        collected the pipe re-emits (carry, collection) — the same pair
        shape, so the channel nests through enclosing composites
        (see core.split_aux for the output doctrine)."""
        key = None
        carry = input
        if boundary_rng:
            key = input.rng                  # missing key fails here, loudly
            carry = input.without('rng')
        new_states, aux = {}, {}
        for nm in names:
            member_in = carry
            if rng_members[nm]:
                key, sub = jax.random.split(key)
                if type(member_in) is not Struct:
                    raise TypeError(f"pipe member '{nm}' consumes rng from its input; "
                                    'the signal into it must be a named Struct to '
                                    'carry the key alongside the data')
                member_in = Struct(**dict(member_in.__items__), rng=sub)
            new_states[nm], out = defs[nm].apply_fn(param[nm], state[nm], member_in)
            carry, member_aux = split_aux(out)
            if member_aux is not None:
                aux[nm] = member_aux
        if aux:
            return Struct(**new_states), (carry, Struct(**aux))
        return Struct(**new_states), carry

    head = defs[names[0]]
    return Serial(
        name='(' + ' >> '.join(names) + ')',
        param_fn=param_fn,
        init_fn=init_fn,
        apply_fn=apply_fn,
        parametric=any(d.parametric for d in defs.values()),
        cyclic=any(d.cyclic for d in defs.values()),
        members=dict(defs),
        given=given,
        rebuild=lambda new: _serial(dict(new), given),
        # a pre-bound member's slot carries its stored param as the slot
        # DEFAULT: the spec states the optionality that binding created
        param_input_spec=_given_defaults(
            hoist_rng({nm: defs[nm].param_input_spec for nm in names}), given)
        if given else None,
        apply_input_spec=(_hoist_apply_rng(head.apply_input_spec)
                          if boundary_rng else head.apply_input_spec),
    )


def _member_init(defs, apply, param, input, rng, seeds):
    """Generated composite init: member states composed by name, one
    rng split toward consumers. When apply is the self-form wiring and
    an input value is given, each member's state is built from the
    input it receives at its own call site — apply threads the carry
    through the wiring in its own call order, so a member whose init
    REQUIRES an input, anywhere in the topology, is served (the
    state-side twin of param discovery). Otherwise members init from
    seeds and rng alone, in declaration order. rng arrives as a raw
    key or a stream (a custom init's declared rng is a KeyStream);
    splitting wants a key, drawn once."""
    if isinstance(rng, KeyStream):
        rng = rng.next()
    if rng is not None and not any('rng' in d.state_input_spec for d in defs.values()):
        raise TypeError('no member init consumes entropy; passing a key is '
                        'presumed an error')
    unknown = set(seeds) - set(defs)
    if unknown:
        raise TypeError(f'unknown composite init members: {sorted(unknown)}')

    authored = _author_call(apply)
    if input is not None and authored is not None:
        call, _ = authored
        w = _InitWired(defs, param, rng, seeds)
        call(w, materialize(input))
        states = w.collect()
        impl = _wrap_apply(apply, defs)
        _check_state_stable(
            states,
            lambda p, s, i: _probe_apply(lambda p_, s_, i_: impl(None, p_, s_, i_), p, s, i),
            param, input)
        return states

    key = rng
    states = {}
    for nm, d in defs.items():
        seed = seeds.get(nm) or Struct()
        if key is not None and 'rng' in d.state_input_spec and 'rng' not in seed:
            key, sub_ = jax.random.split(key)
            seed = seed.replace(rng=sub_)
        try:
            states[nm] = d.build_state(param[nm], seed)
        except TypeError as e:
            raise TypeError(f"member '{nm}': {e}") from e
    return Struct(**states)


def _check_state_stable(states, apply_fn, param, input):
    """A cyclic composite's init state must be shape-stable across one
    step — scan carries it, so a slot that changes shape after a step is
    a latent error (a read-before-fed member that leaked a provisional
    shape into its init, say). Run one step abstractly and compare;
    raise here, named, rather than let it surface as an obscure scan
    carry mismatch downstream."""
    if not jax.tree.leaves(states):
        return
    stepped, _ = jax.eval_shape(lambda s, i: apply_fn(param, s, i), states, materialize(input))
    if _shape_sig(stepped) != _shape_sig(states):
        raise TypeError(
            'composite init state is not shape-stable across a step: a state slot '
            'changes shape after one apply. A member read before it was fed likely '
            'took its shape from its default rather than the wiring — declare that '
            "member's initial shape, or feed it before reading it.")


def composite_init(members: dict[str, NodeDef | Node], apply: Callable, param: Any,
                   *, input: Any = None, rng: Any = None, **seeds: Any) -> Struct:
    """The generated member-state walk as a callable — for a full-custom
    init= that runs the standard walk, then patches (a bus estimator booting
    from the battery's sag curve at charge). members and apply are the
    composite's own; param its constructed params; `seeds` are per-member
    seed bundles (dicts pack into Structs at this boundary)."""
    from nodejax.core import _as_bundle
    defs, _ = _promote_members(members)
    packed = dict(_as_bundle(seeds).__items__)
    return _member_init(defs, apply, param, input, rng, packed)


def _promote_members(members: dict[str, NodeDef | Node]
                     ) -> tuple[dict[str, NodeDef], dict[str, Any]]:
    """Member defs and their stored constructions. A bound Node
    crossing the factory boundary is a transport container: the def
    joins members, its params become the member's construction values
    — used by parameterize wherever kwargs leave the slot open, so
    the member tree is spelled ONCE, at the composition site."""
    defs, given = {}, {}
    for nm, v in members.items():
        if v.bound:
            if jax.tree.leaves(v.param):
                given[nm] = v.param
            v = v.ndef
        defs[nm] = v
    return defs, given


def _shape_sig(x):
    """A pytree's structural shape signature: treedef plus per-leaf shape
    and dtype — for comparing a read's provisional state against the shape
    a later feed defines."""
    leaves, tree = jax.tree.flatten(x)
    return (tree, tuple((jnp.shape(l), jnp.result_type(l)) for l in leaves))


def composite(apply: Callable, *, members: dict[str, GenericDef | NodeDef | Node],
              param: Callable[..., Any] | None = None,
              init: Callable[..., Any] | None = None,
              apply_input_spec: Any = None, name: str | None = None) -> GenericDef | NodeDef | Node:
    """A hand-wired composite over DEF-level members.

    THE HAS-A RULE: Every sub-component relationship in NodeJax MUST be
    registered via `members=dict(...)` (in `composite`, `serial`, or `parallel`).
    Privately closing over sub-nodes in Python variables bypasses NodeJax's
    core mechanisms—disabling static specialization (GenericDef), nested
    parameter/state Struct composition, RNG key splitting, shape inference,
    and structural reflection/surgery (`map_members`, `freeze_by_path`).

    apply is written against self: self.member(x) advances a member, and
    self.param / self.state read the member union Structs.
    """
    if any(isinstance(v, GenericDef) for v in members.values()):
        generic_members = {nm: _as_generic(v) for nm, v in members.items()}

        def specialize_fn(**statics: Any) -> NodeDef | Node:
            unknown = set(statics) - set(generic_members)
            if unknown:
                raise TypeError(f'unknown composite generic members: {sorted(unknown)}')
            specialized = {nm: generic_members[nm].specialize(**statics.get(nm, {}))
                           for nm in generic_members}
            return composite(apply, members=specialized, param=param, init=init,
                             apply_input_spec=apply_input_spec, name=name)

        return GenericDef(name or 'composite(' + ', '.join(members) + ')',
                          specialize_fn, members=generic_members)

    defs, given = _promote_members(members)
    reserved = {'param', 'state', 'collect'} & set(defs)
    if reserved:
        raise TypeError(f'member names shadow self attributes: {sorted(reserved)}')

    authored = _author_call(apply)
    author_fields = authored[1] if authored is not None else None
    declared = apply_input_spec   # apply_input_spec= declares the spec, and
    if declared is None and author_fields is not None:
        # the leaf sugar's rule, shared: a field-style signature IS the
        # spec declaration — fields REQUIRED or carrying their defaults
        declared = _bundle_spec_from_sig(apply, drop=('self',))
    apply_fn = _wrap_apply(apply, defs)   # doubles as init's default input offer
    self_form = authored is not None

    def param_fn(ndef, param_input=Struct()):
        """The composite's param constructor: member slots are
        sub-bundles, a boundary rng splits toward members whose bundles want one.
        When the def carries an input spec and the wiring is self-form, the
        carry threads through apply itself — each member's param built from
        the shape it receives at its own call site (the param-side twin of
        init discovery)."""
        key = param_input.rng if 'rng' in param_input else None
        carry = _input_or_none(ndef)
        slots = {nm: param_input[nm] for nm in defs if nm in param_input}
        if carry is not None and self_form and any(d.param_reads_shape
                                                   for d in defs.values()):
            w = _BuildingWired(defs, given, key, slots)
            _author_call(apply)[0](w, carry)
            for nm, d in defs.items():
                if nm not in w._built:            # members the wiring never called
                    g = slots.get(nm)
                    if g is None and nm in given:
                        w._built[nm] = given[nm]
                    else:
                        w._built[nm], w._key = _member_param(nm, d, g, w._key)
            return Struct(**w._built)
        parts = {}
        for nm, d in defs.items():
            g = slots.get(nm)
            if g is None and nm in given:
                parts[nm] = given[nm]         # stored construction values
            else:
                parts[nm], key = _member_param(nm, d, g, key)
        return Struct(**parts)

    def init_fn(ndef, param, state_input=Struct(), input=None):
        walk = any(d.init_reads_shape or d.init_requires_input for d in defs.values())
        carry = input if input is not None else (_input_or_none(ndef) if walk else None)
        seeds = {nm: state_input[nm] for nm in defs if nm in state_input}
        rng = state_input.rng if 'rng' in state_input else None
        return _member_init(defs, apply, param, carry, rng, seeds)

    # a composite holds no params outside its members: its param tree IS the
    # member union, and an own ctor only chooses how that tree is filled (one
    # value seeding two slots, say). With no parametric member there is no
    # tree for a ctor to fill, so asking for one is an error rather than a
    # composite that quietly grows free params.
    parametric = any(d.parametric for d in defs.values())
    if param is not None and not parametric:
        raise TypeError(
            f"{name or 'composite'}: a param constructor was given, but no member is "
            'parametric; a composite has no params outside its members')

    def _rebuild(new_members):
        # reconstruct from (possibly rewritten) members: apply refers to
        # members through self, so swapping them swaps what self sees; flags
        # recompute from the new members. Stored constructions survive by
        # re-binding their slots (rewriters preserve param meaning)
        members2 = {nm: (Node(d.ndef, given[nm]) if nm in given else d)
                    for nm, d in new_members.items()}
        r = composite(apply, members=members2, param=param, init=init,
                      apply_input_spec=declared, name=name)
        return r.ndef

    ndef = Composite(
        name=name or 'composite(' + ', '.join(defs) + ')',
        param_fn=_lift_param(param) if param is not None else param_fn,
        init_fn=_lift_init(init) if init is not None else init_fn,
        apply_fn=apply_fn,
        parametric=parametric,
        cyclic=any(d.cyclic for d in defs.values()) or init is not None,
        members=dict(defs),
        given=given,
        rebuild=_rebuild,
        apply_input_spec=(_hoist_apply_rng(declared)
                          if any(d.apply_takes_rng for d in defs.values())
                          else declared),
        # an OWN ctor/init defines this composite's bundles; else the
        # member-keyed derivation applies, with stored constructions as slot
        # defaults (None = derivation, nothing stored)
        param_input_spec=_bundle_spec_from_sig(param, drop=('ndef', 'param'))
        if param is not None else (_given_defaults(
            hoist_rng({nm: d.param_input_spec for nm, d in defs.items()}), given)
            if given else None),
        state_input_spec=_state_spec_from_sig(init) if init is not None else None,
    )
    # node_def's convention: non-parametric defs come back bound. The
    # param is the leafless Struct of member slots — the param-side twin
    # of the pipe's leafless state Struct.
    return ndef if parametric else ndef.bind(param_fn(ndef))


def wrapper(apply: Callable, inner: GenericDef | NodeDef | Node, *, name: str | None = None
            ) -> GenericDef | NodeDef | Node:
    """IS-A adaptation of exactly one node: the wrapper's param, init,
    state layout, paths and methods ARE the inner's — flat, a nesting
    level nowhere — and apply wraps the step. The signature names the
    member: apply(<inner>, input) -> output, the first parameter your
    name for the transient step object (call it to step, repeated
    calls chain; .param and .state read the slices).
    """
    if isinstance(inner, GenericDef):
        inner_gen = _as_generic(inner)
        return GenericDef(name or f'wrapper({inner.name})',
                          lambda **kw: wrapper(apply, inner_gen.specialize(**kw), name=name),
                          members=dict(inner=inner_gen))

    nd = inner.ndef
    sig = tuple(inspect.signature(apply).parameters)
    if len(sig) != 2 or sig[1] != 'input' or sig[0] in ('param', 'state', 'input'):
        raise TypeError(f'wrapper apply is (<inner>, input) -> output; got {sig}')

    def apply_fn(_own, param, state, input):
        solo = _Solo(nd, param, state)      # the WRAPPED def, captured
        raw_out = apply(solo, input)
        clean_out, direct_aux = split_aux(raw_out)
        aux_fields = {}
        if direct_aux is not None:
            if isinstance(direct_aux, Struct):
                for k in direct_aux.__keys__:
                    aux_fields[k] = direct_aux[k]
            elif isinstance(direct_aux, dict):
                aux_fields.update(direct_aux)
        if solo._aux:
            aux_fields.update(solo._aux)

        out = (clean_out, Aux(**aux_fields)) if aux_fields else clean_out
        return solo._current, out

    out = nd._replace(name=name or f'wrapper({nd.name})', apply_fn=apply_fn)
    return Node(out, inner.param) if inner.bound else out


def _ident(name: str) -> str:
    """Sanitize a def name into a pipe member identifier."""
    return re.sub(r'\W+', '_', name).strip('_') or 'node'


def _components(x: GenericDef | NodeDef | Node) -> dict[str, GenericDef | NodeDef | Node]:
    """Flatten SERIAL pipes into their named members so (a >> b) >> c
    stays flat; hand-wired composites have members but their wiring is
    free-form, so they enter a pipe atomically."""
    if x.bound:
        if isinstance(x.ndef, Serial):
            return {nm: Node(d, x.param[nm]) for nm, d in x.ndef.members.items()}
        return {_ident(x.ndef.name): x}
    if isinstance(x, Serial):
        return dict(x.members)
    if isinstance(x, GenericDef) and x.members:
        return dict(x.members)
    return {_ident(x.name): x}


def _compose(left: GenericDef | NodeDef | Node,
             right: GenericDef | NodeDef | Node) -> GenericDef | NodeDef | Node:
    """Merge components of both sides (disambiguating duplicate names) into
    one flat serial pipe. Any generic member makes the pipe generic."""
    merged = dict(_components(left))
    for nm, v in _components(right).items():
        final, suffix = nm, 2
        while final in merged:
            final = f'{nm}_{suffix}'
            suffix += 1
        merged[final] = v
    if any(isinstance(v, GenericDef) for v in merged.values()):
        return serial_generic(**{nm: _as_generic(v) for nm, v in merged.items()})
    return serial(**merged)
