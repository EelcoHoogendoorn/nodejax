"""Scoped `self` handles for authored `Composite` and `Wrapper` bodies.

Authored `step(self, ...)` and `init(self, ...)` functions interact with
declared members through a transient `self` handle rather than threading member
parameter and state trees manually. Reading `self.member` snapshots that
member's current view; calling `self.member(...)`, `self.member.scan(...)`,
`self.member.bind(...)`, or `self.member.reset(...)` records the updated state
and any `Aux` output in an invocation-local store.

Three execution scopes implement the same `self` surface: `_Wired` executes a
runtime step, `_BuildingWired` runs the wiring abstractly under `eval_shape` to
discover call-site input shapes before constructing parameters outside the
trace, and `_InitWired` runs `init` or `step` to construct or prime initial
member states in call order. Public `Node`, `PNode`, and `PSNode` views never
enter these transient scopes.
"""

from __future__ import annotations

import inspect
from typing import Callable

import jax
import jax.numpy as jnp
from jax.extend.core import get_opaque_trace_state

from nodejax.core.author_view import AuthorNode
from nodejax.struct import Struct
from nodejax.core.binding import (
    Aux, _UNSET, _bind_method, split_aux,
)
from nodejax.core.definition import Def
from nodejax.core.lifting import _keys_only
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.types import PyTree
from nodejax.tree import tree_first, tree_len

_NO_INPUT = object()


def _trace():
    return get_opaque_trace_state(convention='nodejax')


def _strands(value: PyTree, trace) -> bool:
    """Reports whether `value` carries traced arrays created inside a nested JAX trace."""
    return _trace() != trace and bool(jax.tree.leaves(value))


class _Slot:
    """Routes state writes, `Aux` captures, and child PRNG streams for one runtime member."""
    __slots__ = ('_wired', '_name')

    def __init__(self, wired: _Wired, name: str) -> None:
        self._wired = wired
        self._name = name

    def rng(self, takes_rng: bool) -> MaybeKeyStream:
        return self._wired._rng.child(takes_rng)

    def write(self, state) -> None:
        self._wired._store('state', self._name, state)

    def run(self, contract, param, state_fn: Callable, input, *,
            sequence: bool = False) -> PyTree:
        """Runs `contract.apply`, records the next state and any `Aux`, and returns clean output."""
        new_state, out = contract.apply(
            param, state_fn(), input, self.rng(contract.apply_takes_rng))
        self.write(new_state)
        out, aux = split_aux(out)
        if aux is not None:
            self._wired._store('aux', self._name, aux)
        return out


class _NestedSlot:
    """Routes `bind` and `reset` state writes on a grandchild member back into its parent slot."""
    __slots__ = ('_parent', '_name', '_parent_state_fn', '_shared', '_owner')

    def __init__(self, parent, name: str, parent_state_fn: Callable,
                 shared: bool, owner: str) -> None:
        self._parent = parent
        self._name = name
        self._parent_state_fn = parent_state_fn
        self._shared = shared
        self._owner = owner

    def rng(self, takes_rng: bool) -> MaybeKeyStream:
        return self._parent.rng(takes_rng)

    def write(self, state) -> None:
        if self._shared:
            self._parent.write(state)
        else:
            self._parent.write(
                self._parent_state_fn().replace(**{self._name: state}))

    def run(self, contract, param, state_fn: Callable, input, *,
            sequence: bool = False) -> PyTree:
        raise TypeError(
            f"member {self._name!r} of {self._owner!r} is stepped by the "
            'apply that owns it')


class _Member:
    """Represents a snapshot view of one member inside an authored `step` or `init` scope.

    Calling the handle steps the member from the snapshot state and writes the
    successor state into the enclosing slot; subsequent reads of `self.member`
    observe that updated state, whereas keeping an older `_Member` reference
    retains its earlier state snapshot.
    """
    __slots__ = ('_slot', '_def', '_param', '_state_fn', '_rng_fn')

    def __init__(self, slot, definition: Def, param, state_fn: Callable,
                 rng_fn: Callable | None = None) -> None:
        self._slot = slot
        self._def = definition
        self._param = param
        self._state_fn = state_fn
        self._rng_fn = rng_fn

    def __call__(self, *args: PyTree, **fields: PyTree) -> PyTree:
        from nodejax.core.binding import _bind_call
        return self._slot.run(
            self._def.contract, self._param, self._state_fn,
            _bind_call(self._def, args, fields))

    @property
    def param(self) -> PyTree:
        return self._def.contract._sparse_param(self._param)

    @property
    def state(self) -> PyTree:
        return self._def.contract._sparse_state(self._state_fn())

    def bind(self, param=_UNSET, *, state=_UNSET) -> _Member:
        """Returns a handle with replaced `param` or `state`, writing `state` to the slot."""
        contract = self._def.contract
        bound_param = (self._param if param is _UNSET
                       else contract._dense_param(param))
        if state is _UNSET:
            return _Member(self._slot, self._def, bound_param,
                           self._state_fn, self._rng_fn)
        dense = contract._dense_state(state)
        self._slot.write(dense)
        return _Member(self._slot, self._def, bound_param,
                       lambda: dense, self._rng_fn)

    def reset(self, *args: PyTree, **fields: PyTree) -> _Member:
        """Re-initializes the member's state, stores it in the slot, and returns the rebound handle."""
        from nodejax.core.pnode import PNode
        contract = self._def.contract
        if contract.init_takes_rng:
            fields = {**fields, 'rng': self._slot.rng(True).next()}
        state = PNode(self._def, self._param).init(*args, **fields)
        return self.bind(state=state)

    def scan(self, *args: PyTree, **fields: PyTree) -> PyTree:
        """Scans the member across a sequence, stores the final state, and returns stacked outputs."""
        from nodejax.core.binding import _bind_call
        from nodejax.core.node import Node
        from nodejax.transforms.iteration.scan import scan
        runner = scan(Node(self._def))
        return self._slot.run(
            runner.contract, self._param, self._state_fn,
            _bind_call(runner._def, args, fields), sequence=True)

    def __getattr__(self, name: str):
        methods = self._def.methods
        if methods and name in methods:
            return _bind_method(methods[name],
                                node=lambda: AuthorNode(self._def),
                                param=lambda: self._param,
                                state=self._state_fn,
                                rng=self._rng_fn)
        return _member_of(self._def, name, self._param, self._state_fn,
                          self._rng_fn, self._slot)


def _member_of(definition: Def, name: str, param, state_fn: Callable,
               rng_fn: Callable | None, slot) -> _Member:
    """Resolves a nested member handle, traversing transparent wrappers automatically."""
    transparent = definition.layout.transparent_member
    if name in definition.members:
        member = getattr(definition.members, name)
        if name == transparent:
            slice_param = param
            slice_state = state_fn if member.cyclic else (lambda: ())
        else:
            slice_param = getattr(param, name) if member.parametric else ()
            slice_state = ((lambda: getattr(state_fn(), name))
                           if member.cyclic else (lambda: ()))
        nested = _NestedSlot(
            slot, name, state_fn, name == transparent, definition.name)
        return _Member(nested, member, slice_param, slice_state, rng_fn)
    if transparent is not None:
        return getattr(
            _member_of(definition, transparent, param, state_fn, rng_fn, slot),
            name)
    raise AttributeError(
        f"member node {definition.name!r} has no method or member {name!r}")


class _Wired:
    """Implements the invocation-local `self` scope during a runtime `apply` step."""

    def __init__(self, obj, state, members: Struct, rng: MaybeKeyStream, *,
                 author_rng: bool = False, rng_from: bool | None = None) -> None:
        self._obj = obj
        self._state = state
        self._members = members
        self._new = {}
        self._aux = {}
        self._trace = _trace()
        self._stranded = set()
        self._rng = rng
        # rng_from selects the authored rng value: None narrows to a KeyStream,
        # True forwards the MaybeKeyStream unchanged, and False supplies an empty stream.
        self._boundary = (
            None if not author_rng else
            MaybeKeyStream() if rng_from is False else
            rng if rng_from is True else
            rng._require())

    def sow(self, **kwargs: PyTree) -> None:
        """Records named diagnostic values, losses, or taps in the step's `Aux` output."""
        for k, v in kwargs.items():
            self._store('aux', k, v)

    def _store(self, store: str, name: str, value: PyTree) -> None:
        (self._new if store == 'state' else self._aux)[name] = value
        if _strands(value, self._trace):
            self._stranded.add((store, name))
        else:
            self._stranded.discard((store, name))

    def _settle(self, where: str) -> None:
        """Raises `TypeError` if any state or `Aux` write remained stranded inside a nested trace."""
        if not self._stranded:
            return
        store, name = min(self._stranded)
        if store == 'state':
            raise TypeError(
                f'{where}: member {name!r} stored its state inside a JAX '
                'transformation within the apply (a lax.cond branch, a loop '
                'body, vmap); a value leaves that function only as its '
                f'output. Return self.{name}.state from the function and '
                f'store it with self.{name}.bind(state=...) after it')
        raise TypeError(
            f'{where}: aux {name!r} was produced inside a JAX transformation '
            'within the apply (a lax.cond branch, a loop body, vmap); a value '
            'leaves that function only as its output. Return it from the '
            'function and sow it after it')

    @property
    def __items__(self) -> tuple[tuple[str, _Member], ...]:
        return tuple((name, self.__getattr__(name))
                     for name in self._members.__keys__)

    def __getattr__(self, name: str) -> _Member:
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._members:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        Block = getattr(self._members, name)
        block_param = (getattr(self._obj, name) if Block.parametric else ())
        # Re-reading self.member after a call observes the updated state while
        # earlier _Member handles retain their snapshot.
        current = (self._new[name] if name in self._new
                   else (getattr(self._state, name) if Block.cyclic else ()))
        return _Member(_Slot(self, name), Block, block_param,
                       lambda: current,
                       (lambda: self._boundary) if self._boundary is not None else None)

    def _collect(self) -> Struct | tuple:
        """Assembles the composite's next state, carrying uncalled members unchanged."""
        merged = {name: self._new.get(name, self._state[name])
                  for name in (self._state.__keys__ if self._state != () else ())}
        return Struct(**merged) if merged else ()


class _BuildingMember:
    """Provides the `_Member` surface during abstract parameter-shape discovery."""

    __slots__ = ('_wired', '_name', '_def')

    def __init__(self, wired: _BuildingWired, name: str,
                 definition: Def) -> None:
        self._wired = wired
        self._name = name
        self._def = definition

    def __call__(self, *args: PyTree, **fields: PyTree) -> PyTree:
        from nodejax.core.binding import _bind_call
        return self._wired._call(
            self._name, _bind_call(self._def, args, fields))

    @property
    def param(self) -> PyTree:
        return self._wired._resolved_node(self._name).contract._sparse_param(
            self._wired._ensure_param(self._name))

    @property
    def state(self) -> PyTree:
        return self._wired._resolved_node(self._name).contract._sparse_state(
            self._wired._ensure(self._name))

    def bind(self, param=_UNSET, *, state=_UNSET) -> _BuildingMember:
        return self

    def reset(self, *args: PyTree, **fields: PyTree) -> _BuildingMember:
        return self

    def scan(self, *args: PyTree, **fields: PyTree) -> PyTree:
        from nodejax.core.binding import _bind_call
        from nodejax.core.node import Node
        from nodejax.transforms.iteration.scan import scan
        sequence = _bind_call(scan(Node(self._def))._def, args, fields)
        value = self._wired._call(self._name, tree_first(sequence))
        length = tree_len(sequence)
        return jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (length,) + jnp.shape(leaf)),
            value)

    def __getattr__(self, name: str):
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
        return _member_of(
            self._def, name, self._wired._ensure_param(self._name),
            lambda: self._wired._ensure(self._name),
            lambda: self._wired._boundary,
            _BuildingSlot(self._wired, self._name))


class _InitWired:
    """Implements `self` during state initialization and priming walks.

    On first call to `self.member(x)`, `_InitWired` constructs or primes that
    member's initial state from `x` and records it, while advancing a separate
    working copy on subsequent calls so downstream wiring receives valid outputs.
    When `real` is `False`, the walk runs abstractly to record member input
    shapes and rebuilds cold member states outside the trace.
    """

    def __init__(self, definitions: Struct, param, rng: MaybeKeyStream,
                 member_inputs, state_given: dict, real: bool = True, *,
                 author_rng: bool = False) -> None:
        self._defs = definitions
        self._param = param
        self._real = real
        self._rng = rng
        self._probe = MaybeKeyStream(jax.random.PRNGKey(0))
        self._walk_rng = rng if real else self._probe
        self._member_inputs = member_inputs
        self._init = {}
        self._trace = _trace()
        self._stranded = set()
        self._work = {}
        self._specs = {}
        self._called = set()
        self._given = set()
        for name, state in state_given.items():
            if name in member_inputs:
                state = member_inputs[name]
            self._init[name] = state
            self._work[name] = state
            self._given.add(name)
        self._boundary = (self._walk_rng._require()
                          if author_rng or not real else None)

    def _build(self, name: str, x: PyTree) -> None:
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
            # Record the discovered input spec during an abstract walk; final
            # member states are rebuilt outside the trace in _collect().
            from nodejax.core.spec import spec_of
            self._specs[name] = spec_of(x)
            resolved = d.contract._resolve_def(x, bundled=True)
            child = self._probe.child(resolved.contract.init_takes_rng)
            self._init[name] = resolved.contract.init(
                member_param, member_input, child)
        self._work[name] = self._init[name]
        self._note(name)

    def _note(self, name: str) -> None:
        if _strands(self._init[name], self._trace):
            self._stranded.add(name)
        else:
            self._stranded.discard(name)

    def _ensure(self, name: str) -> PyTree:
        """Builds a cold initial state when `self.member.state` is read before any call."""
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        if name not in self._init:
            self._build(name, _NO_INPUT)
        return self._work[name]

    @property
    def __items__(self) -> tuple[tuple[str, _Member], ...]:
        return tuple((name, self.__getattr__(name))
                     for name in self._defs.__keys__)

    def __getattr__(self, name: str) -> _Member:
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        d = getattr(self._defs, name)
        return _Member(
            _InitSlot(self, name), d,
            (getattr(self._param, name) if d.parametric else ()),
            lambda: self._work[name] if name in self._work else self._ensure(name),
            ((lambda: self._boundary)
             if self._boundary is not None else None))

    def _collect(self) -> Struct | tuple:
        for nm in self._defs.__keys__:
            if nm not in self._init:
                self._build(nm, _NO_INPUT)
        if self._real:
            if self._stranded:
                raise TypeError(
                    f'member {min(self._stranded)!r} took its initial state '
                    'inside a JAX transformation within the composite (a '
                    'lax.cond branch, a loop body, vmap); a value leaves that '
                    'function only as its output. Prime the member outside '
                    'it, in an init written against self')
            states = {nm: self._init[nm]
                      for nm, d in self._defs.__items__ if d.cyclic}
            return Struct(**states) if states else ()
        # Rebuild cold member states outside the abstract trace using the
        # per-member input specs discovered during the walk.
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


class _InitSlot:
    """Records a member's initial state on first touch and advances a working copy thereafter."""
    __slots__ = ('_wired', '_name')

    def __init__(self, wired: _InitWired, name: str) -> None:
        self._wired = wired
        self._name = name

    def rng(self, takes_rng: bool) -> MaybeKeyStream:
        return self._wired._walk_rng.child(takes_rng)

    def write(self, state) -> None:
        wired, name = self._wired, self._name
        if name not in wired._called:
            wired._init[name] = state
            wired._note(name)
            wired._called.add(name)
        wired._work[name] = state

    def run(self, contract, param, state_fn: Callable, input, *,
            sequence: bool = False) -> PyTree:
        wired, name = self._wired, self._name
        element = tree_first(input) if sequence else input
        if name not in wired._called:
            if name not in wired._given:
                wired._build(name, element)
            wired._called.add(name)
        elif not wired._real and name not in wired._specs and name not in wired._given:
            from nodejax.core.spec import spec_of
            wired._specs[name] = spec_of(element)
        from nodejax.core.compose import _probe_apply
        new_state, out = _probe_apply(
            contract.apply, param, wired._work[name], input,
            self.rng(contract.apply_takes_rng))
        wired._work[name] = new_state
        value, aux = split_aux(out)
        return value


class _BuildingSlot:
    """Stores abstract probe states and supplies probe PRNG streams during shape discovery."""
    __slots__ = ('_wired', '_name')

    def __init__(self, wired: _BuildingWired, name: str) -> None:
        self._wired = wired
        self._name = name

    def rng(self, takes_rng: bool) -> MaybeKeyStream:
        return self._wired._probe.child(takes_rng)

    def write(self, state) -> None:
        self._wired._states[self._name] = state


class _BuildingWired:
    """Implements `self` during abstract parameter-shape discovery.

    Each `self.member(x)` call records the input spec `x` arrives with and probes
    the member under `eval_shape` to propagate output shapes downstream; once the
    walk finishes, `parameters()` constructs real member parameters outside the
    abstract trace.
    """

    def __init__(self, definitions: Struct, given, rng: MaybeKeyStream,
                 kwargs: dict, states: dict | None = None) -> None:
        self._defs = definitions
        self._given = given
        self._rng = rng
        self._probe = MaybeKeyStream(jax.random.PRNGKey(0))
        self._boundary = self._probe._require()
        self._kwargs = kwargs
        self._built = {}
        self._states = {} if states is None else dict(states)
        self._specs = {}

    def _resolved_node(self, name: str) -> Def:
        d = getattr(self._defs, name)
        spec = self._specs.get(name, _NO_INPUT)
        return (d if spec is _NO_INPUT else
                d.contract._resolve_def(spec, bundled=True))

    def _ensure_param(self, name: str, input=_NO_INPUT) -> PyTree:
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

    def _ensure(self, name: str, input=_NO_INPUT) -> PyTree:
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

    def _call(self, name: str, input) -> PyTree:
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
        """Constructs concrete member parameters outside `eval_shape` from discovered specs."""
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
    def __items__(self) -> tuple[tuple[str, _BuildingMember], ...]:
        return tuple((name, self.__getattr__(name))
                     for name in self._defs.__keys__)

    def __getattr__(self, name: str) -> _BuildingMember:
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._defs:
            raise TypeError(f"'{name}' is not a member; read data directly off the object")
        return _BuildingMember(self, name, getattr(self._defs, name))


class _RawApply:
    """Wraps an authored `(param, state, input)` step that threads member states directly."""
    __slots__ = ('_apply',)
    fields: tuple[str, ...] = ()
    author_rng = False
    wired = False

    def __init__(self, apply: Callable) -> None:
        self._apply = apply

    def lift(self, definitions: Struct, *,
             rng_from: bool | None = None) -> Callable:
        apply = self._apply
        return lambda definition, p, s, i, rng: apply(p, s, i)


class _WiredApply:
    """Base adapter for an authored `(self, ...)` step that executes through `_Wired`."""
    __slots__ = ('_apply',)
    fields: tuple[str, ...] = ()
    wired = True

    def __init__(self, apply: Callable) -> None:
        self._apply = apply

    @property
    def author_rng(self) -> bool:
        return 'rng' in self.fields

    def run(self, wired, input) -> PyTree:
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
            where = f"{definition.name if definition is not None else 'composite'}.apply"
            out = _keys_only(run(self, input), where)
            self._settle(where)
            clean_out, direct_aux = split_aux(out)
            if direct_aux is not None:
                if type(direct_aux) is Aux:
                    for k in direct_aux.__keys__:
                        self._aux[k] = direct_aux[k]
            new_state = self._collect()
            if self._aux:
                self._aux = _keys_only(self._aux, where)
                return new_state, (clean_out, Aux(**self._aux))
            return new_state, clean_out

        return apply_fn


class _FieldApply(_WiredApply):
    """Unpacks a formed `Struct` input bundle into named arguments on `(self, <fields...>)`."""
    __slots__ = ('fields',)

    def __init__(self, apply: Callable, fields: tuple[str, ...]) -> None:
        super().__init__(apply)
        self.fields = fields

    def run(self, wired, input) -> PyTree:
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
        return self._apply(wired, **kw)


def _authored(apply: Callable) -> _RawApply | _WiredApply:
    """Classifies `apply` as either `(self, <fields...>)` or `(param, state, input)`."""
    sig = tuple(inspect.signature(apply).parameters)
    if sig == ('param', 'state', 'input'):
        return _RawApply(apply)
    if sig[:1] == ('self',) and len(sig) > 1:
        return _FieldApply(apply, sig[1:])
    raise TypeError('composite apply is (self, <fields...>) or the raw '
                    f'(param, state, input) -> (state, output); got {sig}')
