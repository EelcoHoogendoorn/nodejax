"""The `self` sugar: a mutable object interface over a composite's step.

Everything here exists ONLY inside the sugared context of an authored
composite apply/init — self.member(x) advances a member, self.param /
self.state read the live slices, mutation local to the step. The classes
below implement that syntax and its param/init-time twins (shape and state
discovery by running the wiring); _wrap_apply transforms the self-form
function into a raw apply(param, state, input). The NodeDef never sees any
of this: compose's factories consume the transformed functions, and members
are only ever touched through their public contract (build_param /
build_state / apply_fn, with_input resolution).
"""

from __future__ import annotations

import inspect
from functools import partial

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core import NodeDef, split_aux, _as_bundle, _input_or_none, _resolve, _has_rng
from nodejax.authoring import KeyStream

_NO_INPUT = object()   # a read, distinct from a feed of any value (None included)

class _Member:
    """A live member handle on the transient self: calling it steps the
    member (repeated calls chain); attribute access reaches the def's
    methods, bound to the LIVE slices — never a stored construction.
    The param binds always (methods are param-first by contract); a
    second parameter NAMED `state` declares the role and receives the
    member's live state slice — chained, so a read after a step sees
    the advance. Unbound calls through the def pass both explicitly,
    in the same order."""
    __slots__ = ('_call', '_ndef', '_param', '_state_fn')

    def __init__(self, call, ndef, param, state_fn):
        self._call = call
        self._ndef = ndef
        self._param = param
        self._state_fn = state_fn

    def __call__(self, *args, **fields):
        if fields:
            if args:
                raise TypeError('pass ONE input pytree or loose fields, not both')
            return self._call(_as_bundle(fields))
        input, = args
        return self._call(input)

    def __getattr__(self, name):
        methods = self._ndef.methods
        if methods and name in methods:
            fn = methods[name]
            sig = tuple(inspect.signature(fn).parameters)
            if sig[1:2] == ('state',):
                return partial(fn, self._param, self._state_fn())
            return partial(fn, self._param)
        raise AttributeError(f"member def {self._ndef.name!r} has no method {name!r}")


class _Wired:
    """The composite's transient step object — `self` in the reserved
    apply signature. Locally-mutable member-call threader (purity-safe:
    the mutation never escapes the step — the KeyStream precedent)."""

    def __init__(self, obj, state, members, boundary_key=None):
        self._obj = obj
        self._state = state
        self._members = members
        self._new = {}
        self._aux = {}
        self._closed = False
        # the boundary key (peeled off the composite's input bundle when a
        # member consumes apply-rng): split toward member calls, never a
        # shared draw
        self._boundary = KeyStream(boundary_key) if boundary_key is not None else None

    @property
    def param(self):
        """The composite's param Struct: member slots and data reads."""
        return self._obj

    @property
    def state(self):
        """The INCOMING state Struct, for direct reads (a delay
        member's stored value); member calls advance the live slots."""
        return self._state

    @property
    def rng(self) -> KeyStream:
        """The step's KeyStream. With a boundary key (the composite's input
        bundle carried rng), that stream — per-step entropy, split per draw.
        Else the state.rng stream: self.rng.next() yields a fresh key per
        call and the advanced key folds back into state.rng at collect()."""
        if self._boundary is not None:
            return self._boundary
        if '_keystream' not in self.__dict__:
            if not _has_rng(self._state):
                raise TypeError("node state has no 'rng' field to draw from")
            self._keystream = KeyStream(self._state['rng'])
        return self._keystream

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        # the member: a (def, live param slice) pair looked up by name;
        # .ndef normalizes NodeDef and param-less bound Node alike
        if name not in self._members:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        block_def, block_param = self._members[name].ndef, self._obj[name]

        def call(input):
            if self._closed:
                raise TypeError(
                    f"member '{name}' called after collect(); collect reads the "
                    'final member states and CLOSES the step — an advance after '
                    'it could never reach the collected state.')
            # a member that consumes apply-rng draws from the boundary
            # stream unless the wiring routed a key itself — every call an
            # independent draw, explicit keys winning
            if (self._boundary is not None and block_def.apply_takes_rng
                    and 'rng' not in input):
                input = input.replace(rng=self._boundary.next())
            # repeated calls CHAIN: each reads the member's latest state
            # within this step, so calling twice steps twice — integrators
            # accumulate, and rng streams advance (independent draws)
            current = self._new.get(name, self._state[name])
            new_state, out = block_def.apply_fn(block_param, current, input)
            self._new[name] = new_state
            # aux is DIVERTED (core.split_aux doctrine): the wiring
            # receives the clean signal, the collection re-emits at
            # return. Chained calls: the last call's aux stands,
            # matching the state semantics.
            out, member_aux = split_aux(out)
            if member_aux is not None:
                self._aux[name] = member_aux
            return out

        return _Member(call, block_def, block_param,
                       lambda: self._new.get(name, self._state[name]))

    def collect(self, **extra: Any) -> Struct:
        """The composite's new state: original slots, called members at
        their final chained state (repeated calls are sequential steps),
        uncalled members carried unchanged (multi-rate friendly),
        plus any extra wrapper-level fields. Collect CLOSES the step:
        member calls after it raise, because their advances could no
        longer reach the returned state."""
        self._closed = True
        if '_keystream' in self.__dict__:
            self._new['rng'] = self._keystream._key
        merged = {name: self._new.get(name, self._state[name])
                  for name in self._state.__keys__}
        merged.update(extra)
        return Struct(**merged)


class _LazyInitState:
    """`self.state` during an init-time apply run: reading .member (or
    ['member']) yields that member's INITIAL state, built on demand from
    seeds alone — a read is not a feed, so no call-site input is
    offered. Mirrors _Wired.state's incoming-state semantics: the value
    seen is the member's state at step entry, never an advance."""

    def __init__(self, wired):
        self._wired = wired

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._wired._ensure(name)

    def __getitem__(self, name):
        return self._wired._ensure(name)


class _InitWired:
    """`self` for an INIT-time run of the composite apply: on
    self.member(x) it builds that member's INITIAL state from the input
    it receives (offer input=x, rng where the init consumes it), then
    runs the member's apply on that state to produce the output the
    wiring passes downstream. The state-side twin of _BuildingWired:
    each member's state is built from the input it actually sees, in the
    wiring's own call order, so any topology — and any init that
    requires an input — is served. The recorded state is the initial one
    (first touch wins); repeated calls advance a working copy to keep the
    wiring flowing while the recorded initial state stands. self.param
    reads the param Struct, self.state reads member initial states
    (built on demand). Members neither called nor read are init'd from
    seeds alone at collect()."""

    def __init__(self, defs, param, key, seeds):
        self._defs = defs
        self._param = param
        self._key = key
        self._seeds = seeds
        self._init = {}       # recorded initial states
        self._work = {}       # advancing states, for wiring propagation
        self._called = set()  # members whose init came from a feed (authoritative)

    @property
    def param(self):
        return self._param

    @property
    def state(self):
        return _LazyInitState(self)

    def _build(self, name, x):
        """Init member `name` from its seed, plus a call-site input `x`
        when feeding — recording the initial state and seeding the
        working copy."""
        d = self._defs[name]
        seed = (self._seeds.get(name) if self._seeds else None) or Struct()
        if self._key is not None and 'rng' in d.state_input_spec and 'rng' not in seed:
            self._key, sub_ = jax.random.split(self._key)
            seed = seed.replace(rng=sub_)
        if x is _NO_INPUT:
            self._init[name] = d.build_state(self._param[name], seed)
        else:
            self._init[name] = _resolve(d, x).build_state(self._param[name], seed, input=x)
        self._work[name] = self._init[name]

    def _ensure(self, name):
        """The member's initial state for a READ (self.state, a method's
        live slice), built from its seed with no call-site input. For a
        member whose init ignores input (a self-defined start, like a
        battery's charge) this IS its initial state. For a recurrent
        member read before it is fed (a delay in a feedback loop) it is
        the step-zero value; the feed then sets the running shape, and
        the whole init is verified shape-stable across one step (see
        _member_init) — so a leaked provisional cannot pass silently."""
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        if name not in self._init:
            self._build(name, _NO_INPUT)
        return self._work[name]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        d = self._defs[name]

        def call(x):
            if name not in self._called:      # first feed sets the authoritative
                self._build(name, x)          # init, overriding any read default
                self._called.add(name)
            if d.apply_takes_rng and not _has_rng(x):
                x = x.replace(rng=jax.random.PRNGKey(0))   # probe key; discarded
            new_state, out = d.apply_fn(self._param[name], self._work[name], x)
            self._work[name] = new_state
            return split_aux(out)[0]

        return _Member(call, d, self._param[name],
                       lambda: self._work[name] if name in self._work else self._ensure(name))

    def collect(self) -> Struct:
        for nm in self._defs:
            if nm not in self._init:
                self._build(nm, _NO_INPUT)
        return Struct(**{nm: self._init[nm] for nm in self._defs})


class _BuildingWired:
    """`self` for a PARAM-time run of the composite apply: on
    self.member(x) it builds that member's param from the shape x arrives
    with (offer input=x), probes it (init + apply) to get the output to
    pass on, and records the param. The param-side twin of init discovery
    — shapes propagate through the wiring in the wiring's own call order,
    so any topology works. Only member calls are available; a read of
    self.param/self.state (nothing is built yet) is refused."""

    def __init__(self, defs, given, key, kwargs):
        self._defs = defs
        self._given = given
        self._key = key
        self._kwargs = kwargs
        self._built = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}': param-time shape inference sees only member "
                            'calls, not param/state reads — give this composite '
                            'explicit shapes, or omit the input offer')
        d = self._defs[name]

        def call(x):
            from nodejax.compose import _member_param, _probe_seed, _step_carry
            g = self._kwargs.get(name)
            if g is None and name in self._given:
                p = self._given[name]
            else:
                p, self._key = _member_param(name, d, g, self._key, input=x)
            self._built[name] = p
            s = _resolve(d, x).build_state(p, _probe_seed(d), input=x)
            return _step_carry(d, p, s, x, name)

        return call




class _Solo:
    """wrapper's transient step object: the inner node at (param,
    state) — callable to step it, repeated calls chaining; param and
    state readable; the def's methods bound to the live param; the
    advanced state is collected at return."""
    __slots__ = ('_nd', 'param', 'state', '_current')

    def __init__(self, nd, param, state):
        self._nd = nd
        self.param = param
        self.state = state        # incoming, for direct reads
        self._current = state

    def __call__(self, input):
        self._current, out = self._nd.apply_fn(self.param, self._current, input)
        return out

    def __getattr__(self, name):
        methods = self._nd.methods
        if methods and name in methods:
            fn = methods[name]
            sig = tuple(inspect.signature(fn).parameters)
            if sig[1:2] == ('state',):
                return partial(fn, self.param, self._current)
            return partial(fn, self.param)
        raise AttributeError(f"def {self._nd.name!r} has no method {name!r}")


def _wrap_apply(apply: Callable, defs: dict[str, NodeDef]) -> Callable:
    """Transform a composite apply into contract shape, by signature:
    (self, input) builds the transient step object and auto-collects;
    the raw (param, state, input) passes through."""
    sig = tuple(inspect.signature(apply).parameters)
    boundary = any(d.ndef.apply_takes_rng for d in defs.values())
    if sig == ('self', 'input'):
        def apply_fn(p, s, input):
            key = None
            if boundary:
                key = input.rng                  # missing key fails here, loudly
                input = input.without('rng')
            self = _Wired(p, s, defs, boundary_key=key)
            out = apply(self, input)
            new_state = self.collect()
            if self._aux:
                # member aux, diverted under member names, re-emitted as
                # the (output, collection) pair — the channel nests
                return new_state, (out, Struct(**self._aux))
            return new_state, out
        return apply_fn
    if sig == ('param', 'state', 'input'):
        return apply              # the raw triple, author-threaded
    raise TypeError('composite apply is (self, input) -> output, or the raw '
                    f'(param, state, input) -> (state, output); got {sig}')

