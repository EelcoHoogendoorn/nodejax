"""The SUGAR layer: authored functions with human signatures transform into
the contract-shaped functions the NodeDef stores. self, ndef and KeyStream
exist only here.

apply carries one of the four reserved signatures — (input) | (param, input)
| (state, input) | (param, state, input) — and is lifted mechanically into
the uniform apply_fn(param, state, input) -> (state, output) contract.
Richer front-ends (the FOOP keyword DSL, spec-aware declaration) compile
down to node_def calls.
"""

from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable, overload

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import Param, State, ParamFn, ApplyFn
from nodejax.core import (Node, NodeDef, _trivial_init_fn, _trivial_param_fn,
                                REQUIRED, _bundle_spec_from_sig, _has_rng,
                                _METHOD_CHANNELS)


class KeyStream:
    """Locally-mutable key dispenser: next() yields a fresh key per call.
    The mutation is local to the constructor's scope — it never escapes —
    so referential transparency holds (the rewrite.md purity note). Param
    constructors declaring rng receive their offered key pre-wrapped."""

    def __init__(self, key: Any):
        self._key = key

    def next(self) -> Any:
        self._key, sub = jax.random.split(self._key)
        return sub

    def __jax_array__(self) -> Any:
        return self.next()


def _keys_only(tree: Any) -> Any:
    """Collapse any KeyStream leaf to a fresh raw key. Applied at EVERY lift
    exit: KeyStream never escapes the sugared internals — params, states and
    outputs carry raw keys only (an authored `return Struct(rng=rng)` stores
    a key, not a stream)."""
    return jax.tree.map(lambda leaf: leaf.next() if isinstance(leaf, KeyStream) else leaf,
                        tree, is_leaf=lambda x: isinstance(x, KeyStream))


def _as_arrays(tree: Any) -> Any:
    """Cast every leaf of a constructed tree to a jax array. Applied at the
    param and init lift exits, so authored constructors write plain python
    numbers and the stored trees still carry uniform array leaves — the
    per-field jnp.asarray in a constructor is the framework's job, not the
    author's."""
    return jax.tree.map(jnp.asarray, tree)


def _no_var_params(fn: Callable, role: str) -> dict:
    """The natural signature of an authored fn, with *args/**kwargs refused:
    a bundle unpacks by NAME, so an open signature has no meaning."""
    params = inspect.signature(fn).parameters
    if any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
           for p in params.values()):
        raise TypeError(f'{role} takes a closed set of named parameters; *args/**kwargs '
                        'have no meaning against a bundle (declare `param`/`state` to '
                        'receive the whole bundle)')
    return params


def _lift_param(ctor: Callable) -> Callable:
    """Transform an AUTHORED param constructor into the stored impl,
    (ndef, param_input) -> param. The leading ndef is the def's PRIVATE
    binding seam, not the contract: the def exposes this impl publicly as
    param_fn(param_input).

    Reserved names in the authored signature are CHANNELS: `ndef` receives
    the def (shape reflection, resolved by the caller); `rng` receives the
    bundle's key as a KeyStream; `param` receives the whole bundle (a
    dynamically-keyed constructor). Every other name is a bundle field,
    passed when present — the constructor's own defaults fill the rest. The
    bundle is validated against the spec BEFORE this runs (build_param), so
    unknown fields and missing required ones never reach it."""
    params = _no_var_params(ctor, 'a param constructor')
    if 'input' in params:
        raise TypeError('a param constructor reads shape through ndef; '
                        '`input` is not a param-construction channel')
    names = list(params)

    def param_fn(ndef, param_input: Struct = Struct()) -> Param:
        kw: dict = {}
        for nm in names:
            if nm == 'ndef':
                kw[nm] = ndef
            elif nm == 'rng':
                if 'rng' in param_input:
                    kw[nm] = KeyStream(param_input.rng)
            elif nm == 'param':
                kw[nm] = param_input
            elif nm in param_input:
                kw[nm] = param_input[nm]
        return _as_arrays(_keys_only(ctor(**kw)))

    return param_fn


# user apply signature (by parameter names) -> apply_fn.
# 'self' is an alias for 'param': the first slot is THE OBJECT (data and
# collaborator blocks alike; trainability is decided by leaves, not slot
# membership). Leaf nodes that feel like functions write param; composites
# that hold collaborators write self — rewrite.md's original spelling.
_APPLY_LIFTS: dict[tuple[str, ...], Callable[[Callable], ApplyFn]] = {
    ('input',):                   lambda f: lambda p, s, i: (s, f(i)),
    ('param', 'input'):           lambda f: lambda p, s, i: (s, f(p, i)),
    ('self', 'input'):            lambda f: lambda p, s, i: (s, f(p, i)),
    ('state', 'input'):           lambda f: lambda p, s, i: f(s, i),
    ('param', 'state', 'input'):  lambda f: f,
    ('self', 'state', 'input'):   lambda f: f,
}

_OBJECT_NAMES = ('param', 'self')       # first slot: THE OBJECT (param or self)
_APPLY_RESERVED = frozenset(_OBJECT_NAMES) | {'state'}


def _lift_apply_general(apply: Callable) -> ApplyFn:
    """Lift an apply whose TRAILING parameters are input FIELDS: the input
    Struct is unpacked into them by name (a trailing `rng` is promoted to a
    KeyStream). Leading param/self and state come from the offers; a node
    naming the whole `input` uses the fixed lifts. Non-cyclic returns output;
    cyclic returns (state, output)."""
    params = list(inspect.signature(apply).parameters)
    obj = next((p for p in params if p in _OBJECT_NAMES), None)
    has_state = 'state' in params
    fields = [p for p in params if p not in _APPLY_RESERVED]

    def apply_fn(param: Any, state: Any, input: Any) -> Any:
        kw: dict = {}
        if obj is not None:
            kw[obj] = param
        if has_state:
            kw['state'] = state
        for f in fields:
            v = input[f]
            kw[f] = KeyStream(v) if f == 'rng' and not isinstance(v, KeyStream) else v
        out = _keys_only(apply(**kw))
        return out if has_state else (state, out)

    return apply_fn


def _apply_lift(apply: Callable, sig: tuple) -> ApplyFn:
    """Select the apply lift: a fixed whole-`input` pattern, else the general
    unpack when the trailing params are input FIELDS (no whole `input`,
    at least one field; `ndef` in apply is not yet wired)."""
    lift = _APPLY_LIFTS.get(sig)
    if lift is not None:
        return lift
    if 'input' in sig or 'ndef' in sig or not any(p not in _APPLY_RESERVED for p in sig):
        raise TypeError(f'apply must be a reserved signature {list(_APPLY_LIFTS)} or '
                        f'leading param/state then input FIELDS, got {sig}')
    return _lift_apply_general        # curried like _APPLY_LIFTS values: lift(apply) -> apply_fn



def _advance_rng(apply_fn: ApplyFn) -> ApplyFn:
    """Auto-advance the reserved 'rng' state field: each apply consumes a
    fresh key at state.rng and stores its successor, so stochastic cyclic
    nodes never split or thread keys by hand. Keep the field via
    state.replace(...) in user code. No-op for states without the field.

    rng in the OTHER two positions needs no mechanism at all: a key passed
    to a param constructor (parameterize(rng=...)) or carried in the input
    pytree is just data.
    """
    def wrapped(p, s, i):
        if _has_rng(s):
            next_key, use_key = jax.random.split(s.rng)
            new_s, out = apply_fn(p, s.replace(rng=use_key), i)
            return new_s.replace(rng=next_key), out
        return apply_fn(p, s, i)
    return wrapped


def _lift_init(init: Callable[..., State]) -> Callable[..., State]:
    """Transform an AUTHORED init into the stored impl,
    (ndef, param, state_input, input) -> state. The leading ndef is the
    def's PRIVATE binding seam, not the contract: the def exposes this impl
    publicly as init_fn(param, state_input, input=None).

    Reserved names in the authored signature are CHANNELS: `param`/`self`
    receives the built param; `ndef` the def (shape reflection, resolved by
    the caller); `input` a real value of the node's input — when none is
    given, the def's declared spec materializes as the fallback; `rng` the
    seed bundle's key as a KeyStream; `state` the whole seed bundle. Every
    other name is a seed-bundle field, passed when present. A returned
    KeyStream leaf collapses to a fresh key (init(rng): Struct(rng=rng)
    stores a key, not a stream). Bundle validation happens BEFORE this runs
    (build_state)."""
    params = _no_var_params(init, 'an init')
    names = list(params)

    def init_fn(ndef, param: Param, state_input: Struct = Struct(),
                input: Any = None) -> State:
        kw: dict = {}
        for nm in names:
            if nm in _OBJECT_NAMES:
                kw[nm] = param
            elif nm == 'ndef':
                kw[nm] = ndef
            elif nm == 'input':
                if input is not None:
                    kw[nm] = input
                elif ndef.apply_input_spec is not None:
                    from nodejax.spec import materialize
                    kw[nm] = materialize(ndef.apply_input_spec)
                # else: the init's own default, or a loud missing-argument error
            elif nm == 'rng':
                if 'rng' in state_input:
                    kw[nm] = KeyStream(state_input.rng)
            elif nm == 'state':
                kw[nm] = state_input
            elif nm in state_input:
                kw[nm] = state_input[nm]
        return _as_arrays(_keys_only(init(**kw)))

    return init_fn


def _init_requires_input(init: Callable) -> bool:
    """A REQUIRED input (declared, no default) means the init primes from a
    real value — a data need that bubbles to the def's stored
    init_requires_input record, like rng."""
    params = inspect.signature(init).parameters
    return 'input' in params and params['input'].default is inspect.Parameter.empty


def _state_spec_from_sig(init: Callable) -> Struct:
    """The state_input bundle spec, computed by the sugar from the init
    signature and STORED on the def: the seed a caller supplies to init
    BEYOND the param and the apply-input — explicit seed fields plus rng,
    required marked, optional carrying its default. param/self (the built
    param), ndef (encapsulated self-def) and input (the apply-input, its own
    channel) are not part of it."""
    params = inspect.signature(init).parameters
    if 'rng' in params and params['rng'].default is not inspect.Parameter.empty:
        raise TypeError(f'{init.__name__}: rng never has a default; a '
                        'stochastic init requires its key')
    return Struct(**{
        nm: (REQUIRED if p.default is inspect.Parameter.empty else p.default)
        for nm, p in params.items()
        if nm not in ('param', 'self', 'ndef', 'input', 'state')
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)})


# Node/NodeDef attribute names a method may not shadow (real attributes win
# over __getattr__ forwarding, so a colliding method would be unreachable)
_RESERVED_NAMES = frozenset({
    'apply', 'init', 'scan', 'parameterize', 'bind', 'specialize',
    'name', 'parametric', 'cyclic', 'ndef', 'param',
    'param_fn', 'init_fn', 'apply_fn', 'members', 'apply_input_spec', 'methods',
})


def _check_methods(methods: dict[str, Callable] | None) -> dict[str, Callable] | None:
    if methods:
        collisions = set(methods) & _RESERVED_NAMES
        if collisions:
            raise TypeError(f'method names shadow reserved node attributes: {sorted(collisions)}')
        for nm, fn in methods.items():
            _check_method_signature(nm, fn)
    return methods or None


# channel rank in the contract order: binding times — structure, then
# params, then state, then entropy
_CHANNEL_RANK = {'ndef': 0, 'param': 1, 'state': 2, 'rng': 3}


def _check_method_signature(nm: str, fn: Callable) -> None:
    """A method signature reads like every authored signature: the
    channels it declares form a LEADING prefix in the contract order —
    ndef, param, state, rng — followed by the call arguments. self has
    no meaning here: the object channel of a method is param."""
    names = list(_no_var_params(fn, f'method {nm!r}'))
    if 'self' in names:
        raise TypeError(f"method {nm!r}: self has no meaning in a method "
                        "signature; the object channel is param")
    chans = [n for n in names if n in _METHOD_CHANNELS]
    if names[:len(chans)] != chans:
        raise TypeError(f"method {nm!r}: channels {chans} lead the signature; "
                        f"call arguments follow them")
    ranks = [_CHANNEL_RANK[c] for c in chans]
    if ranks != sorted(ranks):
        raise TypeError(f"method {nm!r}: channels follow the contract order "
                        f"(ndef, param, state, rng); got {chans}")


@overload
def node_def(apply: Callable, *, param: ParamFn | None = ...,
             init: Callable[..., State] | None = ..., name: str | None = ...,
             apply_input_spec: Any = ..., methods: dict[str, Callable] | None = ...,
             ) -> NodeDef | Node: ...
@overload
def node_def(apply: None = ..., *, param: ParamFn | None = ...,
             init: Callable[..., State] | None = ..., name: str | None = ...,
             apply_input_spec: Any = ..., methods: dict[str, Callable] | None = ...,
             ) -> Callable[[Callable], NodeDef | Node]: ...
def node_def(apply: Callable | None = None, *, param: ParamFn | None = None,
             init: Callable[..., State] | None = None, name: str | None = None,
             apply_input_spec: Any = None, methods: dict[str, Callable] | None = None,
             ) -> NodeDef | Node | Callable[[Callable], NodeDef | Node]:
    """Author a node from functions with human signatures — the sugar frontend
    that computes the def's specs and transforms the fns into contract shape
    (see core).

    apply is required: one of the fixed whole-`input` signatures
    (input) | (param, input) | (state, input) | (param, state, input), or
    leading param/self and state followed by trailing INPUT FIELDS, which the
    lift unpacks from the input Struct by name (a trailing rng arrives as a
    KeyStream).

    param is the constructor of the node's param bundle: its signature IS
    the declaration — each parameter a bundle field (required when it has no
    default), published as param_input_spec. A parametric apply REQUIRES a
    param constructor; an apply reading a field no constructor declares is
    broken by definition. init is required iff apply takes state; its
    signature declares the seed bundle the same way (state_input_spec).

    Reserved names in a ctor/init signature are CHANNELS, never bundle
    fields: `param`/`self` (the built param, init only), `ndef` (the def,
    for shape reflection — zeros_like(ndef.input)), `input` (init only: a
    real value of the node's input, wiring-supplied), `rng` (the bundle's
    key, delivered as a KeyStream — draw via rng.next(), no split
    bookkeeping), `state` (the whole seed bundle, for dynamically-keyed
    recipes). *args/**kwargs are definition-time errors: a bundle unpacks by
    name, so an open signature has no meaning.

    Entropy never vanishes: rng appears in the bundle spec iff the signature
    declares it, so a key passed to a deterministic node fails as an
    ordinary unknown bundle field.

    apply_input_spec optionally declares the input spec (a pytree of
    ShapeDtypeStruct and/or concrete values, see spec.py); shape-generic
    nodes leave it None and bind one later (with_input, or a wiring that
    resolves it). methods attaches non-reserved callables whose reserved
    parameter names are CHANNELS, injected by the view that binds the
    method, a leading prefix in the contract order: ndef (the def),
    param (the object; self has no meaning in a method), state (the
    live state), rng (the boundary key stream) — the same names, the
    same meaning as in every authored signature. Every other parameter
    follows as a call argument; a channel a view cannot offer (state on
    a bare Node) is the caller's to pass by keyword.

    Returns a NodeDef when parametric (bind via parameterize), and an
    already-bound Node otherwise. Usable as a decorator for apply-only nodes.
    """
    if apply is None:
        # decorator-with-arguments: @node_def(name=..., apply_input_spec=...)
        # evaluates node_def WITHOUT the apply first; the partial then
        # receives the decorated function as its `apply`
        return partial(node_def, param=param, init=init, name=name,
                       apply_input_spec=apply_input_spec, methods=methods)

    sig = tuple(inspect.signature(apply).parameters)
    lift = _apply_lift(apply, sig)

    parametric = any(n in sig for n in _OBJECT_NAMES)
    cyclic = 'state' in sig

    if param is not None and not parametric:
        raise TypeError('a param constructor was given, but apply does not take param')
    if parametric and param is None:
        raise TypeError(f"'{name or apply.__name__}': apply takes param but declares no "
                        'param constructor. A parametric node declares its param bundle '
                        'with param=; there is no default that fabricates one from kwargs.')
    if cyclic and init is None:
        raise TypeError('apply takes state; an init function is required')
    if init is not None and not cyclic:
        raise TypeError('an init function was given, but apply does not take state')

    ndef = NodeDef(
        name=name or apply.__name__,
        param_fn=_lift_param(param) if parametric else _trivial_param_fn,
        init_fn=_lift_init(init) if cyclic else _trivial_init_fn,
        apply_fn=_advance_rng(lift(apply)) if cyclic else lift(apply),
        parametric=parametric,
        cyclic=cyclic,
        # declared spec wins; a field-style apply otherwise publishes its
        # signature-derived marker bundle (fields + rng), resolved at binding
        apply_input_spec=apply_input_spec if apply_input_spec is not None
        else (_bundle_spec_from_sig(apply, drop=tuple(_APPLY_RESERVED))
              if sig not in _APPLY_LIFTS and 'input' not in sig else None),
        methods=_check_methods(methods),
        init_requires_input=_init_requires_input(init) if cyclic else False,
        param_reads_shape=(parametric and param is not None
                           and 'ndef' in inspect.signature(param).parameters),
        init_reads_shape=(cyclic and bool({'ndef', 'input'}
                                          & set(inspect.signature(init).parameters))),
        # the sugar computes the IN specs from the natural signatures and the
        # container stores them (ndef/input are injected channels, not fields)
        param_input_spec=_bundle_spec_from_sig(param, drop=('ndef', 'param'))
        if parametric else None,
        state_input_spec=_state_spec_from_sig(init) if cyclic else None,
    )
    return ndef if parametric else Node(ndef, ())


def derive(parent: NodeDef | Node, *, apply: Callable | None = None,
           init: Callable[..., State] | None = None, param: ParamFn | None = None,
           name: str | None = None, apply_input_spec: Any = None,
           state_input_spec: Any = None,
           methods: dict[str, Callable] | None = None) -> NodeDef | Node:
    """Derive a new def from an existing one — functional record update in
    place of subclassing. Everything not overridden is inherited; overrides
    carry natural signatures and are lifted exactly like node_def's.
    Methods merge, child winning per name. No hierarchy, no MRO: 'super' is
    explicit — an overriding apply calls Parent.apply_fn(param, state, input)
    in closure.

    Flags recompute from the effective pieces, so derivation can move a
    node through the lattice: overriding apply with a state-taking
    signature (plus an init) makes a cyclic node from a plain parent;
    dropping param from the signature makes it non-parametric (inherited
    param/init that the new apply cannot use are discarded).

    Returns a NodeDef when parametric, a bound Node otherwise. A bound
    parent's params are NOT carried over — the param layout may have
    changed; rebind explicitly via .bind or parameterize.
    """
    pd = parent.ndef

    if apply is not None:
        sig = tuple(inspect.signature(apply).parameters)
        lift = _apply_lift(apply, sig)
        parametric = any(n in sig for n in _OBJECT_NAMES)
        cyclic = 'state' in sig
        apply_fn = _advance_rng(lift(apply)) if cyclic else lift(apply)
    else:
        parametric, cyclic = pd.parametric, pd.cyclic
        apply_fn = pd.apply_fn

    if param is not None and not parametric:
        raise TypeError('a param constructor was given, but the derived apply does not take param')
    if init is not None and not cyclic:
        raise TypeError('an init function was given, but the derived apply does not take state')
    if cyclic and init is None and not pd.cyclic:
        raise TypeError('the derived apply takes state; supply an init (parent has none)')
    if parametric and param is None and not pd.parametric:
        raise TypeError(f"'{name or pd.name}': derived apply takes param but neither an "
                        'override nor the parent declares a param constructor.')

    ndef = NodeDef(
        name=name or pd.name,
        param_fn=(_lift_param(param) if param is not None else pd._param_impl)
        if parametric else _trivial_param_fn,
        init_fn=(_lift_init(init) if init is not None else pd._init_impl)
        if cyclic else _trivial_init_fn,
        apply_fn=apply_fn,
        parametric=parametric,
        cyclic=cyclic,
        apply_input_spec=apply_input_spec if apply_input_spec is not None
        else ((_bundle_spec_from_sig(apply, drop=tuple(_APPLY_RESERVED))
               if sig not in _APPLY_LIFTS and 'input' not in sig else None)
              if apply is not None else pd.apply_input_spec),
        methods=_check_methods({**(pd.methods or {}), **(methods or {})}),
        init_requires_input=(_init_requires_input(init) if init is not None
                             else pd.init_requires_input) if cyclic else False,
        param_reads_shape=(('ndef' in inspect.signature(param).parameters)
                           if param is not None else pd.param_reads_shape)
        if parametric else False,
        init_reads_shape=(bool({'ndef', 'input'} & set(inspect.signature(init).parameters))
                          if init is not None else pd.init_reads_shape) if cyclic else False,
        param_input_spec=(_bundle_spec_from_sig(param, drop=('ndef', 'param'))
                          if param is not None else pd.param_input_spec)
        if parametric else None,
        # an explicitly declared seed spec wins over sig derivation: a
        # wrapper COMPUTES its spec from what it wraps (boundary hoist),
        # which no single authored signature can state
        state_input_spec=(state_input_spec if state_input_spec is not None
                          else _state_spec_from_sig(init) if init is not None
                          else pd.state_input_spec) if cyclic else None,
    )
    return ndef if parametric else Node(ndef, ())
