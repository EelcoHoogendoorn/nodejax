"""The `self` sugar: a mutable object interface over a composite's step.

Everything here exists ONLY inside the sugared context of an authored
composite apply/init — self.member(x) advances a member, self.param /
self.state read the live slices, mutation local to the step. The classes
below implement that syntax and its param/init-time twins (shape and state
discovery by running the wiring); the authored apply object lowers the
self-form function to a canonical call over raw Def values. Public Node
views never enter this machinery.
"""

from __future__ import annotations

import inspect

import jax

from nodejax.core.author_view import AuthorNode
from nodejax.struct import Struct
from nodejax.core.binding import (
    Aux, _bind_method, split_aux,
)
from nodejax.core.definition import Def
from nodejax.core.lifting import _keys_only
from nodejax.core.rng import MaybeKeyStream

_NO_INPUT = object()   # a read, distinct from a feed of any value (None included)


class _Member:
    """A live member handle on the transient self: calling it steps the
    member (repeated calls chain); attribute access reaches the node's
    methods, slot-bound to the LIVE slices — never a stored
    construction. The reserved parameter names are the slots
    (core._bind_method): node is the member's def, param its param,
    state its chained state slice (a read after a step sees the
    advance), rng the wiring's boundary stream. Unbound calls through the node
    pass the slots explicitly."""
    __slots__ = ('_call', '_def', '_param', '_state_fn', '_rng_fn')

    def __init__(self, call, definition: Def, param, state_fn, rng_fn=None):
        self._call = call
        self._def = definition
        self._param = param
        self._state_fn = state_fn
        self._rng_fn = rng_fn

    def __call__(self, *args, **fields):
        from nodejax.core.binding import (_bind_call)
        return self._call(_bind_call(self._def, args, fields))

    def __getattr__(self, name):
        methods = self._def.methods
        if methods and name in methods:
            return _bind_method(methods[name],
                                node=lambda: AuthorNode(self._def),
                                param=lambda: self._param,
                                state=self._state_fn,
                                rng=self._rng_fn)
        raise AttributeError(
            f"member node {self._def.name!r} has no method {name!r}")


class _Wired:
    """The composite's transient step object — `self` in the reserved
    apply signature. Its state and RNG cursors are invocation-local."""

    def __init__(self, obj, state, members, rng: MaybeKeyStream, *,
                 author_rng: bool = False, rng_from: bool | None = None):
        self._obj = obj
        self._state = state
        self._members = members
        self._new = {}
        self._aux = {}
        self._rng = rng
        # rng_from decides what the authored rng argument is: None narrows
        # the invocation stream to a KeyStream, True forwards the stream
        # as it is, keyed or empty, and False hands over an empty one.
        self._boundary = (
            None if not author_rng else
            MaybeKeyStream() if rng_from is False else
            rng if rng_from is True else
            rng._require())

    @property
    def param(self):
        """The composite's param Struct: member slots and data reads."""
        return self._obj

    @property
    def state(self):
        """The INCOMING state Struct, for direct reads (a delay
        member's stored value); member calls advance the live slots."""
        return self._state

    def sow(self, **kwargs: Any) -> None:
        """Sow auxiliary values (taps, losses, activity) into the step's aux stream."""
        for k, v in kwargs.items():
            self._aux[k] = v

    @property
    def __items__(self):
        """Member handles as (name, handle) pairs, mirroring Struct."""
        return tuple((name, self.__getattr__(name))
                     for name in self._members.__keys__)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        # the member: a (def, live param slice) pair looked up by name;
        if name not in self._members:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        Block = getattr(self._members, name)
        block_param = (getattr(self._obj, name) if Block.parametric else ())

        def call(input):
            # repeated calls CHAIN: each reads the member's latest state
            # within this step, so calling twice steps twice — integrators
            # accumulate, and rng streams advance (independent draws)
            current = (self._new[name] if name in self._new
                       else (getattr(self._state, name)
                             if Block.cyclic else ()))
            child = self._rng.child(Block.contract.apply_takes_rng)
            new_state, out = Block.contract.apply(block_param, current, input, child)
            self._new[name] = new_state
            # aux DIVERTS (core.split_aux): the wiring gets the clean
            # signal; chained calls keep the LAST call's aux, like state
            out, member_aux = split_aux(out)
            if member_aux is not None:
                self._aux[name] = member_aux
            return out

        return _Member(call, Block, block_param,
                       lambda: (self._new[name] if name in self._new
                                else (getattr(self._state, name)
                                      if Block.cyclic else ())),
                       (lambda: self._boundary) if self._boundary is not None else None)

    def _collect(self) -> Struct:
        """The composite's new state: original slots, called members at
        their final chained state (repeated calls are sequential steps),
        and uncalled members carried unchanged (multi-rate friendly)."""
        merged = {name: self._new.get(name, self._state[name])
                  for name in (self._state.__keys__ if self._state != () else ())}
        return Struct(**merged) if merged else ()


class _LazyInitState:
    """`self.state` during an init-time apply run: reading .member (or
    ['member']) yields that member's INITIAL state, built on demand from
    its own bundle alone — a read is not a feed, so no call-site input is
    supplied. Mirrors _Wired.state's incoming-state semantics: the value
    seen is the member's state at step entry, never an advance."""

    def __init__(self, wired):
        self._wired = wired

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._wired._ensure(name)

    def __getitem__(self, name):
        return self._wired._ensure(name)


class _LazyBuildingParam:
    """``self.param`` during the parameter-shape walk.

    A wiring may use one member's parameters to form another member's input.
    Construct the requested member slot on demand. A constructor that truly
    needs its call-site shape still has to be called before it can be read.
    """

    def __init__(self, wired):
        self._wired = wired

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._wired._ensure_param(name)

    def __getitem__(self, name):
        return self._wired._ensure_param(name)


class _BuildingMember:
    """A member handle for the parameter-shape walk.

    It has the same two surfaces as runtime ``_Member``: calling records the
    call-site shape and advances throwaway state, while attribute access binds
    custom methods to lazily constructed parameter/state slices. Returning a
    plain function here used to erase methods during parameterization.
    """

    __slots__ = ('_wired', '_name', '_def')

    def __init__(self, wired, name, definition: Def):
        self._wired = wired
        self._name = name
        self._def = definition

    def __call__(self, *args, **fields):
        from nodejax.core.binding import _bind_call
        return self._wired._call(
            self._name, _bind_call(self._def, args, fields))

    def __getattr__(self, name):
        methods = self._def.methods
        if methods and name in methods:
            return _bind_method(
                methods[name],
                node=lambda: AuthorNode(
                    self._wired._resolved_node(self._name)),
                param=lambda: self._wired._ensure_param(self._name),
                state=lambda: self._wired._ensure(self._name),
                rng=lambda: self._wired._boundary,
            )
        raise AttributeError(
            f"member node {self._def.name!r} has no method {name!r}")


class _InitWired:
    """`self` for an INIT-time run of the composite apply: on
    self.member(x) it builds that member's INITIAL state from the input
    it receives (pass input=x, rng where the init consumes it), then
    runs the member's apply on that state to produce the output the
    wiring passes downstream. The state-side twin of _BuildingWired:
    each member's state is built from the input it actually sees, in the
    wiring's own call order, so any topology — and any init that
    requires an input — is served. The recorded state is the initial one
    (first touch wins); repeated calls advance a working copy to keep the
    wiring flowing while the recorded initial state stands. self.param
    reads the param Struct, self.state reads member initial states
    (built on demand). Members neither called nor read are init'd from
    their own bundles alone when the state is finalized.

    `real` records whether the TOP-level input was data or the zeros of a
    resolved shape. It is deliberately not forwarded to members: what a
    member receives here is a value the wiring COMPUTED, and a source (a
    bus voltage, a constant) produces a genuine one whatever built the
    walk. That is how a member deep in a topology gets something to prime
    from at all (see test_actuator's 48 V bus). The zeros' reach is
    bounded instead at the transforms, which do not compute."""

    def __init__(self, definitions, param, rng: MaybeKeyStream, member_inputs,
                 state_given, real=True, *, author_rng: bool = False):
        self._defs = definitions
        self._param = param
        self._real = real     # data at the top, or zeros of a shape (see class doc)
        self._rng = rng
        self._probe = MaybeKeyStream(jax.random.PRNGKey(0))
        self._walk_rng = rng if real else self._probe
        self._member_inputs = member_inputs
        self._init = {}       # recorded initial states
        self._work = {}       # advancing states, for wiring propagation
        self._specs = {}      # spec walk: the shape each member was fed
        self._called = set()  # members whose init came from a feed (authoritative)
        self._given = set()
        from nodejax.core.binding import _has_rng_deep
        from nodejax.core.compose import _rekeyed
        for name, state in state_given.items():
            if name in member_inputs:
                state = member_inputs[name]
            if _has_rng_deep(state_given[name]):
                state = _rekeyed(
                    state, rng.next(),
                    f"member '{name}'")
            self._init[name] = state
            self._work[name] = state
            self._given.add(name)
        # Abstract walks use private entropy; their values are discarded.
        self._boundary = (self._walk_rng._require()
                          if author_rng or not real else None)

    @property
    def param(self):
        return self._param

    @property
    def state(self):
        return _LazyInitState(self)

    def _build(self, name, x):
        """Init member `name` from its own bundle, plus a call-site input `x`
        when feeding — recording the initial state and starting the
        working copy."""
        d = getattr(self._defs, name)
        member_input = (getattr(self._member_inputs, name)
                        if self._member_inputs else Struct())
        member_param = (getattr(self._param, name) if d.parametric else ())
        if x is _NO_INPUT:
            child = self._walk_rng.child(d.contract.init_takes_rng)
            self._init[name] = d.contract.init(
                member_param, member_input, child)
        elif self._real:
            child = self._rng.child(d.contract.init_takes_rng)
            self._init[name] = d.contract.prime(
                member_param, member_input, d.contract.intake(x), child)
        else:
            # the abstract walk: record the shape fed, build a THROWAWAY
            # tracer state to keep the walk moving; finalization rebuilds outside
            from nodejax.core.spec import spec_of
            self._specs[name] = spec_of(x)
            resolved = d.contract._resolve_def(x, bundled=True)
            child = self._probe.child(resolved.contract.init_takes_rng)
            self._init[name] = resolved.contract.init(
                member_param, member_input, child)
        self._work[name] = self._init[name]

    def _ensure(self, name):
        """The member's initial state for a READ (self.state, a method's
        live slice), built from its own bundle with no call-site input. For a
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

    @property
    def __items__(self):
        """Member handles as (name, handle) pairs, mirroring Struct."""
        return tuple((name, self.__getattr__(name))
                     for name in self._defs.__keys__)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        d = getattr(self._defs, name)

        def call(x):
            if name not in self._called:      # first feed sets the authoritative
                if name not in self._given:
                    self._build(name, x)      # init, overriding any read default
                self._called.add(name)
            from nodejax.core.compose import _probe_apply
            child = self._walk_rng.child(d.contract.apply_takes_rng)

            new_state, out = _probe_apply(
                d.contract.apply,
                (getattr(self._param, name) if d.parametric else ()),
                self._work[name], x, child)
            self._work[name] = new_state
            value, aux = split_aux(out)
            return value

        return _Member(
                       call, d,
                       (getattr(self._param, name) if d.parametric else ()),
                       lambda: self._work[name] if name in self._work else self._ensure(name),
                       ((lambda: self._boundary)
                        if self._boundary is not None else None))

    def _collect(self) -> Struct:
        for nm in self._defs.__keys__:
            if nm not in self._init:
                self._build(nm, _NO_INPUT)
        if self._real:
            states = {nm: self._init[nm]
                      for nm, d in self._defs.__items__ if d.cyclic}
            return Struct(**states) if states else ()
        # the abstract walk built only tracers; rebuild outside it, each
        # member at the shape it was fed, from its own bundle alone
        states = {}
        for nm, d in self._defs.__items__:
            if nm in self._given:
                states[nm] = self._init[nm]
                continue
            member_input = (getattr(self._member_inputs, nm)
                            if self._member_inputs else Struct())
            spec = self._specs.get(nm)
            resolved = (d.contract._resolve_def(spec, bundled=True)
                        if spec is not None else d)
            child = self._rng.child(resolved.contract.init_takes_rng)
            s = resolved.contract.init(
                (getattr(self._param, nm) if d.parametric else ()),
                member_input, child)
            if d.cyclic:
                states[nm] = s
        return Struct(**states) if states else ()


class _BuildingWired:
    """`self` for a PARAM-time run of the composite apply: on
    self.member(x) it builds that member's param from the shape x arrives
    with (pass input=x), probes it (init + apply) to get the output to
    pass on, and records the param. The param-side twin of init discovery
    — shapes propagate through the wiring in the wiring's own call order,
    so any topology works. Member methods and direct ``self.param`` /
    ``self.state`` reads use the same lazily built throwaway slices as member
    calls, preserving the authored runtime surface during discovery."""

    def __init__(self, definitions, given, rng: MaybeKeyStream, kwargs):
        self._defs = definitions
        self._given = given
        self._rng = rng
        self._probe = MaybeKeyStream(jax.random.PRNGKey(0))
        self._boundary = self._probe._require()
        self._kwargs = kwargs
        self._built = {}
        self._states = {}
        self._specs = {}      # the shape each member was fed, for rebuilding outside

    @property
    def param(self):
        return _LazyBuildingParam(self)

    @property
    def state(self):
        return _LazyInitState(self)

    def _resolved_node(self, name):
        """The member definition at the shape known so far in this walk."""
        d = getattr(self._defs, name)
        spec = self._specs.get(name, _NO_INPUT)
        return (d if spec is _NO_INPUT else
                d.contract._resolve_def(spec, bundled=True))

    def _ensure_param(self, name, input=_NO_INPUT):
        """Construct one live probe-param slot, at most once."""
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        if input is not _NO_INPUT:
            from nodejax.core.spec import spec_of
            self._specs[name] = spec_of(input)
        if name in self._built:
            return self._built[name]

        d = getattr(self._defs, name)
        if not d.parametric:
            self._built[name] = ()
            return ()
        g = self._kwargs.get(name)
        if name in self._given:
            p = self._kwargs[name]
        else:
            from nodejax.core.compose import _member_param
            input_spec = None if input is _NO_INPUT else input
            try:
                p = _member_param(
                    name, d, g,
                    self._probe.child(d.contract.param_takes_rng),
                    input_spec=input_spec, bundled=True)
            except TypeError as e:
                if input is _NO_INPUT and d.calls.param.reads_def:
                    raise TypeError(
                        f"member '{name}' reads its parameters before a call "
                        'establishes the input shape they require') from e
                raise
        self._built[name] = p
        return p

    def _ensure(self, name, input=_NO_INPUT):
        """Build the throwaway initial state used by reads and methods."""
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        if name in self._states:
            return self._states[name]
        d = getattr(self._defs, name)
        if not d.cyclic:
            return ()
        p = self._ensure_param(name, input)
        resolved = self._resolved_node(name)
        if resolved.contract.init_requires_input and input is _NO_INPUT:
            raise TypeError(
                f"member '{name}' reads its state before a call establishes "
                'the input needed to initialize it')
        try:
            # This state exists only inside eval_shape and is discarded.  If
            # the member primes from its input, the abstract call-site value
            # is therefore the right probe: it supplies shape flow without
            # ever becoming persisted initialization data.
            if input is _NO_INPUT:
                state = resolved.contract.init(
                    p, Struct(),
                    self._probe.child(resolved.contract.init_takes_rng))
            else:
                state = resolved.contract.prime(
                    p, Struct(), resolved.contract.intake(input),
                    self._probe.child(resolved.contract.init_takes_rng))
        except TypeError as e:
            if input is _NO_INPUT and d.calls.init.reads_def:
                raise TypeError(
                    f"member '{name}' reads its state before a call establishes "
                    'the input needed to initialize it') from e
            raise
        self._states[name] = state
        return state

    def _call(self, name, input):
        """Probe one member call and retain its discovered call-site shape."""
        d = getattr(self._defs, name)
        p = self._ensure_param(name, input)
        state = self._ensure(name, input)
        resolved = self._resolved_node(name)
        from nodejax.core.compose import _probe_apply
        try:
            new_state, out = _probe_apply(
                resolved.contract.apply, p, state, input,
                self._probe.child(resolved.contract.apply_takes_rng))
        except Exception as e:
            raise TypeError(f"walk failed at member '{name}': {e}") from e
        self._states[name] = new_state
        value, aux = split_aux(out)
        return value

    def parameters(self) -> Struct:
        """The params, built OUTSIDE the walk.

        The walk runs the wiring abstractly to discover the shape each member
        is fed, so everything constructed inside it is a tracer. What comes out
        is therefore the SHAPES, and the params are built from them here, where
        nothing is traced. Members the wiring never called are built from their
        own bundles, in declaration order."""
        from nodejax.core.compose import _member_param
        out = {}
        for name, d in self._defs.__items__:
            if not d.parametric:
                continue
            g = self._kwargs.get(name)
            if name in self._given:
                out[name] = self._kwargs[name]
            else:
                out[name] = _member_param(
                    name, d, g,
                    self._rng.child(d.contract.param_takes_rng),
                    input_spec=self._specs.get(name), bundled=True)
        return Struct(**out)

    @property
    def __items__(self):
        """Member handles as (name, handle) pairs, mirroring Struct."""
        return tuple((name, self.__getattr__(name))
                     for name in self._defs.__keys__)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        return _BuildingMember(self, name, getattr(self._defs, name))


class _RawApply:
    """An apply authored as the contract triple itself: (param, state, input).

    There is no wiring object and nothing to discover — the author threads
    member states by hand. It declares no input fields, so it declares no
    spec, and the construction walks have no wiring to run."""
    __slots__ = ('_apply',)
    fields: tuple[str, ...] = ()
    author_rng = False
    wired = False

    def __init__(self, apply: Callable):
        self._apply = apply

    def lift(self, definitions: Struct, *,
             rng_from: bool | None = None) -> Callable:
        apply = self._apply
        return lambda definition, p, s, i, rng: apply(p, s, i)


class _WiredApply:
    """An apply authored against `self`: the wiring builds the transient step
    object, and the same call drives the param- and init-time discovery runs.

    ``scope`` is the author's view of each walk object; None hands the walk
    object over as it is. Subclasses differ only in how the input slot
    reaches the author."""
    __slots__ = ('_apply', '_scope')
    fields: tuple[str, ...] = ()
    wired = True

    def __init__(self, apply: Callable, scope: Callable | None = None):
        self._apply = apply
        self._scope = scope

    @property
    def author_rng(self) -> bool:
        return 'rng' in self.fields

    def run(self, wired, input):
        raise NotImplementedError

    def lift(self, definitions: Struct, *,
             rng_from: bool | None = None) -> Callable:
        author_rng = self.author_rng
        run = self.run

        def apply_fn(definition, p, s, input, rng):
            members = (definitions if definition is None else
                       definition.members)
            self = _Wired(
                p, s, members, rng, author_rng=author_rng, rng_from=rng_from)
            out = _keys_only(
                run(self, input),
                f"{definition.name if definition is not None else 'composite'}.apply")
            clean_out, direct_aux = split_aux(out)
            if direct_aux is not None:
                if type(direct_aux) is Aux:
                    for k in direct_aux.__keys__:
                        self._aux[k] = direct_aux[k]
            new_state = self._collect()
            if self._aux:
                self._aux = _keys_only(
                    self._aux,
                    f"{definition.name if definition is not None else 'composite'}.apply")
                # member aux & self.sow(...) re-emitted as the (output, collection) pair
                return new_state, (clean_out, Aux(**self._aux))
            return new_state, clean_out

        return apply_fn


class _FieldApply(_WiredApply):
    """(self, <fields...>): the input bundle unpacked by name, so the
    signature IS the input spec declaration. A declared rng channel and all
    member calls share the invocation's ordered RNG cursor."""
    __slots__ = ('fields',)

    def __init__(self, apply: Callable, fields: tuple[str, ...],
                 scope: Callable | None = None):
        super().__init__(apply, scope)
        self.fields = fields

    def run(self, wired, input):
        if not issubclass(type(input), Struct):
            raise TypeError('authored composite expects a formed input bundle')
        kw = {}
        for f in self.fields:
            if f == 'rng':
                if wired._boundary is None:
                    raise TypeError(
                        'the authored apply runs while init threads a real '
                        'input and requires an RNG key; pass rng= to init')
                kw[f] = wired._boundary
            else:
                kw[f] = input[f]
        view = wired if self._scope is None else self._scope(wired)
        return self._apply(view, **kw)


def _authored(apply: Callable, *,
              scope: Callable | None = None) -> _RawApply | _WiredApply:
    """Which of the three authored apply forms this is, as an object that
    answers for itself: how to run the wiring, what fields it declares, and
    how to lift it into the contract impl. ``scope`` is the author's view of
    each walk object, when the plain walk object is not it."""
    sig = tuple(inspect.signature(apply).parameters)
    if sig == ('param', 'state', 'input'):
        return _RawApply(apply)          # the contract form: the bundle whole
    if sig[:1] == ('self',) and len(sig) > 1:
        # trailing names are the input fields, `input` one like any other
        return _FieldApply(apply, sig[1:], scope)
    raise TypeError('composite apply is (self, <fields...>) or the raw '
                    f'(param, state, input) -> (state, output); got {sig}')
