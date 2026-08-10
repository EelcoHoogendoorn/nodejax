"""The contract core: NodeDef (blueprint) and Node (bound pytree instance).

Every node is three pure functions against one uniform BUNDLED contract:

    param_fn(param_input)                    -> param pytree
    init_fn(param, state_input, input=None)  -> state pytree
    apply_fn(param, state, input)            -> (state, output)

A bundle is a pytree of real values — the declared input arguments of the fn
it feeds, as fields — validated against the def's stored IN spec: unknown
fields and missing REQUIRED fields are loud errors. rng rides a bundle as a
raw jax key. `input` at init is its own channel — a real value of the node's
input, typed by apply_input_spec. Nothing reserved appears anywhere in this
interface: self, ndef and KeyStream exist only inside the sugared node
internals — the stored impls receive their (resolved) def through a private
binding seam, and the public accessors above are what everyone consumes.
build_param / build_state are the validated entries; parameterize packs
loose kwargs into a bundle as the one public sugar.

This module is the only layer that executes. Everything else in the package
constructs (authoring, generic) or rewrites (transforms, compose) these
objects. CONTRACTS OPERATE STRICTLY ON CONTRACTS: node transforms,
composition, and tree surgery (batch, scan, map_members, tree_freeze) operate
on NodeDefs strictly via their 3 contract functions (param_fn, init_fn,
apply_fn), never on Python signatures, self bindings, or authoring sugar
parameters. Relying exclusively on the 3-function contract is the core
guarantee that enables transforms and compositions to be 100% general over
arbitrary models. Specs are stored metadata published by the producer that
built the fns; the OUT specs are derived by eval_shape (spec.meta), and
nothing consults specs at apply time.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable, TYPE_CHECKING

import jax

from nodejax.struct import Struct
from nodejax.types import Param, State, Input, Output, PyTree, ParamFn, InitFn, ApplyFn

if TYPE_CHECKING:
    from nodejax.generic import GenericDef


class _Required:
    """Marker for a required input-bundle field: a constructor parameter with
    no default, whose value the caller must supply. Its shape/dtype is the
    spec's concern; this is the requirement marker."""
    def __repr__(self) -> str:
        return 'REQUIRED'


REQUIRED = _Required()


def _bundle_spec_from_sig(fn: Callable, *, drop: tuple = ()) -> Struct:
    """The input-bundle spec of a lifted fn, read from the underlying
    constructor signature (@wraps lets inspect follow through the lift): each
    declared field maps to its default value, or REQUIRED when it has none.
    Names in `drop` are omitted — the encapsulated `ndef`, the leading
    object/state slots. rng stays: it is a bundle field like any other."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return Struct()
    out = {}
    for nm, p in params.items():
        if nm in drop or p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD):
            continue
        if nm == 'rng' and p.default is not inspect.Parameter.empty:
            raise TypeError(f'{fn.__name__}: rng never has a default; a '
                            'stochastic fn requires its key')
        out[nm] = REQUIRED if p.default is inspect.Parameter.empty else p.default
    return Struct(**out)


def hoist_rng(subs: dict[str, Any]) -> Struct:
    """Compose member bundle specs into the composite's OWN bundle spec. The
    composite has an rng requirement iff any member does, recorded once at its
    top level: the caller passes one key, and the composite splits it toward
    every member that declares the need. The member slots in the composite's
    bundle carry the per-member input fields the caller supplies — keys are
    the composite's to forward, so rng is not among them. Member defs and
    their own specs are untouched; this only derives the composite's. A
    deterministic composite has no rng field at all, so a passed key fails as
    an ordinary unknown field.

    rng is a PER-FUNCTION concern and this helper is per-bundle by
    construction: it composes ONE function's bundle family per call — the
    members' param specs (from Composite.param_input_spec) or their state
    specs (from Composite.state_input_spec) — so param and init each get
    their own boundary rng, keyed and split independently at their own call
    time. The apply side's twin is _hoist_apply_rng."""
    hoist = False
    out = {}
    for nm, s in subs.items():
        if _has_rng(s):
            hoist = True
            s = s.without('rng')
        out[nm] = s
    if not hoist:
        return Struct(**out)
    return Struct(rng=REQUIRED, **out)   # rng never has a default


def _split_rng(bundle: 'Struct', count: int) -> tuple[Any, 'Struct']:
    """Peel the boundary key off an input bundle and split it toward `count`
    consumers: (keys, data) — one independent stream each, the data fields
    untouched. The write-side of the rng doctrine: one key at the boundary,
    never a shared draw."""
    return jax.random.split(bundle.rng, count), bundle.without('rng')


def _with_rng(data: 'Struct', key: Any) -> 'Struct':
    """One consumer's input: the shared data fields plus its own key."""
    return data.replace(rng=key)


def _has_rng(tree: Any) -> bool:
    """Whether a bundle-shaped tree (a spec or a runtime value) carries an
    rng field — the ONE home for that discrimination."""
    return isinstance(tree, Struct) and 'rng' in tree


def _hoist_apply_rng(spec: Any) -> Any:
    """A composite whose members consume apply-rng advertises ONE boundary
    rng field on its own apply spec. A marker spec gains rng as REQUIRED; a
    RESOLVED spec gains a key-shaped leaf, staying resolved (the walks
    materialize a zero key to probe with); an inexpressible spec (None, or
    a bare non-Struct tree) stays as it is — routing still works at apply,
    read off the member specs."""
    if spec is None or not isinstance(spec, Struct) or 'rng' in spec:   # early-outs incl. the one home's shape
        return spec
    if _spec_resolved(spec):
        from nodejax.spec import spec_of
        return Struct(rng=spec_of(jax.random.PRNGKey(0)), **dict(spec.__items__))
    return Struct(rng=REQUIRED, **dict(spec.__items__))


def _spec_resolved(spec: Any) -> bool:
    """Whether an apply input spec is RESOLVED (shapes/values, materializable)
    rather than a signature-derived marker bundle awaiting binding. None is
    not resolved; a tree containing a REQUIRED marker is not resolved."""
    if spec is None:
        return False
    return not any(leaf is REQUIRED for leaf in jax.tree.leaves(spec))


def _spec_sig(tree: Any) -> tuple:
    """A pytree's structural shape signature (treedef + per-leaf shape and
    dtype), taken through spec_of so concrete and spec leaves compare
    alike. For validating one shape against another."""
    from nodejax.spec import spec_of
    leaves, treedef = jax.tree.flatten(spec_of(tree))
    return (treedef, tuple((tuple(l.shape), l.dtype) for l in leaves))


def _resolve(nd: 'NodeDef', value: Any) -> 'NodeDef':
    """Resolve a def's input spec against a value by the uniform rule:
    fill the spec from `spec_of(value)` when the def has no RESOLVED spec
    (none declared, or only a signature-derived marker bundle), or
    validate the value against a resolved spec. Bundles validate by the
    bundle rule, the same one every other bundle boundary applies:
    unknown fields are a NAMED conflict, a present field must match the
    spec's shape, and a spec field absent from the value is OPTIONAL —
    its concrete spec value is the default the apply unpack fills.
    Non-bundle trees compare whole. Returns the def with a resolved
    spec."""
    spec = nd.apply_input_spec
    if not _spec_resolved(spec):
        return nd.with_input(value)
    if isinstance(spec, Struct) and isinstance(value, Struct):
        unknown = set(value.__keys__) - set(spec.__keys__)
        if unknown:
            raise TypeError(f"{nd.name}: unknown input fields {sorted(unknown)}; "
                            f"the spec fields are {sorted(spec.__keys__)}")
        for k in value.__keys__:
            if _spec_sig(spec[k]) != _spec_sig(value[k]):
                raise TypeError(
                    f"{nd.name}.{k}: input shape {_spec_sig(value[k])[1]} conflicts "
                    f"with its declared spec {_spec_sig(spec[k])[1]}")
        return nd
    if _spec_sig(spec) != _spec_sig(value):
        raise TypeError(
            f"{nd.name}: input shape {_spec_sig(value)[1]} conflicts with its "
            f"declared spec {_spec_sig(spec)[1]}")
    return nd


def split_aux(output: Output) -> tuple[Output, Any]:
    """The aux channel: clean output and auxiliary data separation.

    Returns (clean_output, aux).
    aux is non-None when output is a 2-tuple `(clean_output, aux_data)`
    where `aux_data` is an Aux or Struct instance (sown losses, metrics, taps).
    Positional 2-tuples returning raw arrays or lists pass through as clean output data.
    """
    from nodejax.struct import Struct, Aux
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], (Struct, Aux)):
        return output[0], output[1]
    return output, None


# === trivial components ===

def _trivial_param_fn(ndef: Any, param_input: Any = None) -> Param:
    """param_fn of a non-parametric node: the empty param."""
    return ()


def _trivial_init_fn(ndef: Any, param: Param, state_input: Any = None,
                     input: Any = None) -> State:
    """init_fn of a non-cyclic node: the empty state."""
    return ()


def _as_bundle(fields: dict) -> 'Struct':
    """The ONE public-boundary sugar: loose kwargs pack into a bundle.
    Sub-bundles are Structs. Everything below the boundary speaks bundles."""
    return Struct(**fields)


# names that are CHANNELS in a method signature, injected by the view
# that binds the method
_METHOD_CHANNELS = frozenset({'param', 'state', 'ndef', 'rng'})


def _bind_method(fn: Callable, offers: dict[str, Callable]) -> Callable:
    """Bind a def method to a view. Reserved parameter names are CHANNELS
    — ndef (the def), param (the object), state (the live state), rng
    (the boundary key stream) — the same names with the same meaning as
    in every authored signature, declared as a leading prefix in that
    order (validated at authoring, _check_method_signature); every other
    parameter is a call argument, filled positionally or by keyword. offers maps channel
    name to a zero-arg supplier, read at CALL time, so a state read
    after a member step sees the advance. A channel the view does not
    offer is the caller's to pass by keyword, and an explicit keyword
    always beats injection. rng arrives in the method as a KeyStream
    either way: a view offers the boundary stream itself, an explicit
    keyword passes a key that is wrapped at this seam — one drawing
    idiom (rng.next()) in every context."""
    names = list(inspect.signature(fn).parameters)
    fields = [n for n in names if n not in _METHOD_CHANNELS]
    channels = [n for n in names if n in _METHOD_CHANNELS]

    def bound(*args: Any, **kwargs: Any) -> Any:
        if len(args) > len(fields):
            raise TypeError(f'{fn.__name__}() takes {len(fields)} call argument(s) '
                            f'{fields}; the reserved names {channels} are injected')
        kw = dict(zip(fields, args))
        doubled = kw.keys() & kwargs.keys()
        if doubled:
            raise TypeError(f'{fn.__name__}() got multiple values for {sorted(doubled)}')
        kw.update(kwargs)
        for nm in channels:
            if nm in kw:
                if nm == 'rng':
                    from nodejax.authoring import KeyStream
                    kw[nm] = KeyStream(kw[nm])   # an explicit key, wrapped at the seam
                continue
            supplier = offers.get(nm)
            if supplier is not None:
                kw[nm] = supplier()
        return fn(**kw)

    return bound


# === the form ===

class NodeDef:
    """A node blueprint in contract form.

    Not a pytree: defs are program structure, bound Nodes are data.
    Transforms rewrite defs (def -> def); binding produces instances
    (def -> Node)."""

    name: str
    param_fn: ParamFn
    init_fn: InitFn
    apply_fn: ApplyFn
    parametric: bool
    cyclic: bool
    apply_input_spec: Any | None
    methods: dict[str, Callable] | None
    tags: frozenset[str]

    def __init__(self, name: str, param_fn: ParamFn, init_fn: InitFn, apply_fn: ApplyFn,
                 parametric: bool, cyclic: bool,
                 apply_input_spec: Any | None = None, methods: dict[str, Callable] | None = None,
                 init_requires_input: bool = False, param_reads_shape: bool = False,
                 init_reads_shape: bool = False,
                 param_input_spec: Any | None = None, state_input_spec: Any | None = None,
                 tags: frozenset[str] = frozenset()):
        self.name = name
        # the IMPL fns, PRIVATE: param/init impls are def-first ((ndef, ...))
        # so the binding seam can hand any impl its own resolved def. self,
        # ndef and KeyStream exist only inside the sugared node internals —
        # the public accessors below expose the pure contract signatures.
        self._param_impl = param_fn
        self._init_impl = init_fn
        self._apply_impl = apply_fn
        self.parametric = parametric
        self.cyclic = cyclic
        self.apply_input_spec = apply_input_spec  # declared input (pytree of ShapeDtypeStruct); None = shape-generic
        self.methods = methods  # callables whose reserved parameter names are CHANNELS (_bind_method)
        # init PRIMES from a real input value (a value of apply_input_spec —
        # its own channel, never a state_input field). Recorded at authoring
        # from the init signature; composite factories bubble it from members.
        self.init_requires_input = init_requires_input
        # param construction READS the def's input spec (a ctor naming ndef):
        # not a bundle field (the channel is encapsulated), so this stored
        # record is THE record — it gates a composite's shape-discovery walk.
        self.param_reads_shape = param_reads_shape
        # init READS the def's input spec or consumes an input value (a sig
        # naming ndef or input): gates a composite's init-time carry walk.
        self.init_reads_shape = init_reads_shape
        # the IN bundle specs, STORED: the producer that builds the fns
        # computes them (the sugar from signatures, a factory from members, a
        # transform from its inner def). None = not published; the properties
        # fall back to transitional derivation until every producer publishes.
        self._param_input_spec = param_input_spec
        self._state_input_spec = state_input_spec
        self.tags = frozenset(tags)

    def __getattr__(self, name: str) -> Callable:
        """Unbound method access: the raw function, channels explicit."""
        methods = object.__getattribute__(self, 'methods')
        if methods and name in methods:
            return methods[name]
        raise AttributeError(f'NodeDef {self.name!r} has no attribute {name!r}')

    # --- the public contract fns: bundles in, nothing reserved anywhere ---
    @property
    def param_fn(self) -> Callable:
        """param_fn(param_input) -> param — the contract fn. The stored impl
        is unbound, (ndef, param_input); reading this accessor binds THIS def
        JIT (Python's own method mechanism), so a resolved copy of the def
        binds itself. Prefer build_param, which validates the bundle first."""
        impl, this = self._param_impl, self

        def param_fn(param_input: Struct = Struct()) -> Param:
            return impl(this, param_input)
        return param_fn

    @property
    def init_fn(self) -> Callable:
        """init_fn(param, state_input, input=None) -> state — the contract
        fn, bound JIT from the unbound impl exactly like param_fn. Prefer
        build_state, which validates."""
        impl, this = self._init_impl, self

        def init_fn(param: Param, state_input: Struct = Struct(),
                    input: Any = None) -> State:
            return impl(this, param, state_input, input)
        return init_fn

    @property
    def apply_fn(self) -> Callable:
        """apply_fn(param, state, input) -> (state, output) — the contract
        fn, bound JIT from the unbound impl exactly like param_fn and
        init_fn. The stored impl is (ndef, param, state, input): a
        composite resolves its members from the def it is handed, so
        swapping a member needs no re-lift and the canonical form is
        closed under member substitution."""
        impl, this = self._apply_impl, self

        def apply_fn(param: Param, state: State, input: Input) -> tuple[State, Output]:
            return impl(this, param, state, input)
        return apply_fn

    @property
    def ndef(self) -> NodeDef:
        """The def itself — uniform access at either binding stage:
        x.ndef is the def whether x is a NodeDef or a bound Node."""
        return self

    @property
    def bound(self) -> bool:
        """Whether this is a bound node (a def carries no param): the
        binding-stage question, answered by the type."""
        return False

    @property
    def resolved(self) -> bool:
        """Whether this def's input SHAPE is known — declared, bound by
        with_input, or filled by a wiring. The question the walks ask before
        reading `input`; asking it is not the same as having a value."""
        return _spec_resolved(self.apply_input_spec)

    @property
    def input(self) -> Any:
        """Convenience shape-reflection sugar.

        Returns a dummy zero-filled PyTree (`jnp.zeros(shape, dtype)`) matching
        the node's resolved `apply_input_spec`.

        CRITICAL SEMANTICS:
        - This property exists strictly as authoring sugar so constructors can write
          `jnp.zeros_like(ndef.input)` or `jnp.ones_like(ndef.input)`.
        - `ndef` carries ONLY static blueprint metadata and holds no real data.
        - Zero arrays produced by `ndef.input` must NEVER be passed internally into
          channels or arguments expecting real numerical data (`input`).
        """
        if not _spec_resolved(self.apply_input_spec):
            raise TypeError(
                f"{self.name}: reads its input shape (ndef.input) but no shape is "
                'resolved — declare it (apply_input_spec=), bind with_input(...), or '
                'place the node in a wiring that resolves it')
        from nodejax.spec import materialize
        return materialize(self.apply_input_spec)

    @property
    def apply_takes_rng(self) -> bool:
        """Whether apply consumes entropy from its input bundle — read off
        the apply input spec ('rng' among its fields)."""
        return _has_rng(self.apply_input_spec)

    @property
    def param_input_spec(self) -> Any:
        """The bundle a caller supplies to param_fn — the rigid IN contract:
        ctor fields (required marked, optional carrying its default; rng
        included, ndef excluded — encapsulated). Nonparametric -> (). STORED:
        the producer that builds the fns publishes the spec; an unpublished
        parametric def is a construction error surfaced at first read."""
        if not self.parametric:
            return ()
        if self._param_input_spec is None:
            raise TypeError(f'{self.name}: parametric but no param_input_spec was '
                            'published — the producer of this def must supply it')
        return self._param_input_spec

    @property
    def state_input_spec(self) -> Any:
        """The bundle a caller supplies to init_fn — the state seed: explicit
        seed fields plus rng (required marked, optional carrying its default);
        param, ndef and the apply-input are excluded. Non-cyclic -> ().
        STORED by the producer, like param_input_spec."""
        if not self.cyclic:
            return ()
        if self._state_input_spec is None:
            raise TypeError(f'{self.name}: cyclic but no state_input_spec was '
                            'published — the producer of this def must supply it')
        return self._state_input_spec

    # --- the contract: one bundle in, validated against the spec ---
    def build_param(self, param_input: Struct = Struct()) -> Param:
        """Construct params from ONE bundle — the contract entry.
        The bundle is validated against param_input_spec: unknown fields and
        missing REQUIRED fields are loud errors. rng rides the bundle as a
        raw jax key. A constructor that reads shape does so through its def
        (this def, or a with_input-resolved copy) — never a bundle field."""
        spec = self.param_input_spec
        given = set(param_input.__keys__)
        if spec == ():
            if given:
                raise TypeError(f'{self.name} is not parametric; its param bundle is empty')
            return self._param_impl(self, Struct())  # a composite's empty param is its member Struct
        unknown = given - set(spec.__keys__)
        if unknown:
            raise TypeError(f'{self.name}.build_param: unknown bundle fields '
                            f'{sorted(unknown)}; the bundle is {sorted(spec.__keys__)}')
        missing = [k for k in spec.__keys__ if spec[k] is REQUIRED and k not in given]
        if missing:
            raise TypeError(f'{self.name}.build_param: missing required bundle fields {missing}')
        return self._param_impl(self, param_input)

    def build_state(self, param: Param, state_input: Struct = Struct(),
                    input: Any = None) -> State:
        """Construct state from the param, `state_input` Struct, and optionally a
        real value of the node's INPUT — the contract entry. The
        Struct is validated against state_input_spec (rng rides it as a raw
        key). `input` is its own channel, typed by apply_input_spec, never a
        state_input field: the node's own input, supplied by the wiring (a
        composite's walk, scan's first element) or explicitly here, priming
        or shaping the state — and resolving the def's spec on the way in.

        A non-cyclic node has no state to build: its empty state comes back,
        and an offered input is simply unused — wirings offer the carry
        uniformly, without asking each member whether it is stateful. Seed
        FIELDS to a non-cyclic node remain an error: they claim state that
        does not exist."""
        spec = self.state_input_spec
        given = set(state_input.__keys__)
        if spec == ():
            if given:
                raise TypeError(f'{self.name} is not cyclic; its state bundle is empty')
            return self._init_impl(self, param)
        unknown = given - set(spec.__keys__)
        if unknown:
            raise TypeError(f'{self.name}.build_state: unknown bundle fields '
                            f'{sorted(unknown)}; the bundle is {sorted(spec.__keys__)}')
        missing = [k for k in spec.__keys__ if spec[k] is REQUIRED and k not in given]
        if missing:
            raise TypeError(f'{self.name}.build_state: missing required bundle fields {missing}')
        # an input-priming init is satisfiable two ways: a real value
        # passed here (or threaded by a wiring), or a RESOLVED spec the
        # init lift materializes as the fallback; only a def with
        # neither must refuse
        if (input is None and self.init_requires_input
                and not _spec_resolved(self.apply_input_spec)):
            raise TypeError(f'{self.name} primes its state from a real input value; '
                            'pass input=<value> or declare an input spec')
        # a given input value resolves the def on the way in (fill or
        # validate), so a shape-reading init sees a resolved ndef
        nd = self if input is None else _resolve(self, input)
        return nd._init_impl(nd, param, state_input, input)

    # --- binding ---
    def parameterize(self, param_input: Struct | None = None, /, **fields) -> Node:
        """Construct params and bind them into a Node: ONE bundle, or loose
        fields packed into one — the single public sugar."""
        if param_input is not None and fields:
            raise TypeError('pass ONE param bundle or loose fields, not both')
        bundle = param_input if param_input is not None else _as_bundle(fields)
        return Node(self, self.build_param(bundle))

    def bind(self, param: Param) -> Node:
        """Bind an already-constructed param pytree."""
        return Node(self, param)

    def _replace(self, **changes: Any) -> NodeDef:
        """A copy of this def with fields overridden, preserving the concrete
        type — a leaf stays a NodeDef, a composite stays a Composite (with its
        members). The single copy path; no mutation."""
        fields = dict(name=self.name, param_fn=self._param_impl, init_fn=self._init_impl,
                      apply_fn=self._apply_impl, parametric=self.parametric,
                      cyclic=self.cyclic, apply_input_spec=self.apply_input_spec,
                      methods=self.methods, init_requires_input=self.init_requires_input,
                      param_reads_shape=self.param_reads_shape,
                      init_reads_shape=self.init_reads_shape,
                      param_input_spec=self._param_input_spec,
                      state_input_spec=self._state_input_spec,
                      tags=self.tags)
        fields.update(changes)
        return type(self)(**fields)

    def with_input(self, input: Any) -> NodeDef:
        """A copy of this def with its input spec bound to `input` — a
        spec or a concrete value alike; a concrete tree is reduced to
        its spec (spec_of), so an arbitrarily large array handed in is
        never stored, only its shape. A new def; no mutation."""
        from nodejax.spec import spec_of
        return self._replace(apply_input_spec=spec_of(input))

    def __call__(self, *args: Any, **kwargs: Any) -> Node:
        """Shorthand for parameterize."""
        return self.parameterize(*args, **kwargs)

    # --- composition ---
    def __rshift__(self, other: GenericDef | NodeDef | Node) -> GenericDef | NodeDef | Node:
        """Serial composition; pipes stay flat: (a >> b) >> c == a >> b >> c."""
        from nodejax.compose import _compose
        return _compose(self, other)

    # --- the unbound mirror of the Node surface: param explicit, first ---
    # binding STORES the param; the unbound surface PASSES it — the same
    # call otherwise, sugar included: d.apply(param, ...) == d.bind(param).apply(...)
    def apply(self, param: Param, *args: Any, **fields: Any) -> Any:
        """Node.apply with the param explicit and first."""
        return Node(self, param).apply(*args, **fields)

    def init(self, param: Param, state_input: Any = None, /, *,
             input: Any = None, **fields: Any) -> State:
        """Node.init with the param explicit and first."""
        return Node(self, param).init(state_input, input=input, **fields)

    def scan(self, param: Param, state: State | None, inputs: Input) -> tuple[State, Output]:
        """Node.scan with the param explicit and first."""
        return Node(self, param).scan(state, inputs)

    @property
    def selectable(self) -> dict[str, NodeDef]:
        """The members a name-based selection can address. A leaf has none."""
        return {}

    def map_leaves(self, fn: Callable[[NodeDef], Any]) -> Any:
        """Map fn(leaf) over leaf defs under this node, constructing a Struct tree that
        mirrors the state pytree hierarchy for JAX transformation primitives (e.g. jax.vmap).

        Example:
        >>> state_in = map_node_leaves(net, lambda m: None if 'single_batch_state' in m.tags else 0)
        >>> # state_in is Struct(linear=0, norm=0), passed directly to jax.vmap in_axes/out_axes:
        >>> jax.vmap(apply_fn, in_axes=(None, state_in, 0), out_axes=(state_in, 0))(param, state, input)
        """
        return fn(self) if self.cyclic else 0

    def map_state(self, state: Any, fn: Callable[[NodeDef, Any], Any]) -> Any:
        """Map fn(leaf_def, leaf_state) -> new_leaf_state over matching def and state trees.

        Example:
        >>> # Transform or inspect state leaves based on matching node properties:
        >>> clean_state = map_state_leaves(net, state, lambda d, s: jax.tree.map(jnp.zeros_like, s))
        >>> # Returns a Struct tree matching state's layout with transformed leaf values
        """
        return fn(self, state) if self.cyclic else state

    def __repr__(self) -> str:
        tags = 'P' * self.parametric + 'C' * self.cyclic
        return f'NodeDef({self.name}{":" + tags if tags else ""})'


class Composite(NodeDef):
    """A node constructed from named member nodes. Its NODE CONTRACT is identical
    to any other node (the same param_fn/init_fn/apply_fn/specs/methods); Composite
    adds only what the REWRITE layer needs — the member defs and a rebuild constructor."""

    def __init__(self, name: str, param_fn: ParamFn, init_fn: InitFn, apply_fn: ApplyFn,
                 parametric: bool, cyclic: bool, members: dict[str, NodeDef],
                 given: dict[str, Any] = {},
                 apply_input_spec: Any | None = None, methods: dict[str, Callable] | None = None,
                 rebuild: Callable | None = None,
                 init_requires_input: bool | None = None,
                 param_reads_shape: bool | None = None, init_reads_shape: bool | None = None,
                 param_input_spec: Any | None = None, state_input_spec: Any | None = None,
                 tags: frozenset[str] = frozenset()):
        if init_requires_input is None:      # derived from members; _replace preserves
            init_requires_input = any(d.init_requires_input for d in members.values())
        if param_reads_shape is None:
            param_reads_shape = any(d.param_reads_shape for d in members.values())
        if init_reads_shape is None:
            init_reads_shape = any(d.init_reads_shape for d in members.values())
        super().__init__(name, param_fn, init_fn, apply_fn, parametric, cyclic,
                         apply_input_spec, methods, init_requires_input=init_requires_input,
                         param_reads_shape=param_reads_shape, init_reads_shape=init_reads_shape,
                         param_input_spec=param_input_spec, state_input_spec=state_input_spec,
                         tags=tags)
        self.members = members
        self.given = given
        self.rebuild = rebuild

    @property
    def param_input_spec(self) -> Any:
        if not self.parametric:
            return ()
        if self._param_input_spec is not None:
            return self._param_input_spec
        return hoist_rng({nm: d.param_input_spec for nm, d in self.members.items()})

    @property
    def state_input_spec(self) -> Any:
        if not self.cyclic:
            return ()
        if self._state_input_spec is not None:
            return self._state_input_spec
        return hoist_rng({nm: d.state_input_spec for nm, d in self.members.items()})

    @property
    def selectable(self) -> dict[str, NodeDef]:
        return self.members

    def map_leaves(self, fn: Callable[[NodeDef], Any]) -> Any:
        """Map fn(leaf_node) -> value over member leaves, constructing a Struct
        tree of state metadata (e.g. vmap axes) matching member state hierarchy.

        Returns 0 if the composite has no state (cyclic=False). Non-cyclic member
        subtrees (e.g. frozen state slots) return 0 as the vmap axis spec for `()`."""
        if not self.cyclic:
            return 0
        return Struct(**{nm: m.map_leaves(fn) for nm, m in self.members.items()})

    def map_state(self, state: Any, fn: Callable[[NodeDef, Any], Any]) -> Any:
        """Map fn(leaf_def, leaf_state) -> new_leaf_state over matching member and
        state subtrees, returning a Struct tree of transformed state values."""
        if not self.cyclic:
            return state
        return Struct(**{nm: m.map_state(state[nm], fn)
                         for nm, m in self.members.items() if nm in state})

    def _replace(self, **changes: Any) -> NodeDef:
        base = dict(members=self.members, given=self.given, rebuild=self.rebuild)
        base.update(changes)
        return super()._replace(**base)


class Serial(Composite):
    """A sequential pipeline composite where member outputs chain into subsequent inputs."""
    pass


class Wrapper(Composite):
    """A node produced by a TRANSFORM of one other node: a composite with
    exactly one member, `inner`.

    Being a Composite is what makes it REWRITABLE — map_members, tree_freeze
    and tree_detach reach it through the same rebuild recipe every composite
    carries, with no special case at those call sites. A transform supplies
    that recipe as a closure over its own arguments, since only the call site
    knows them: rebuild={'inner': d} -> batch(d, n, axis).

    It is TRANSPARENT in both directions that would otherwise leak the extra
    level. The contract fns are the transform's own, so param and state keep
    the wrapped node's shape and no 'inner' key appears in any tree; the
    walks below skip straight to the wrapped def; and member lookup answers
    with the wrapped node's members, so selecting a member by name passes
    through the tower rather than finding only 'inner'.

    Without it a wrapper is indistinguishable from a leaf and a walk stops at
    the transform, answering a per-leaf question (which state slots a member
    wants batched, which member a rewrite selects) once for the whole tower.
    """

    def __init__(self, *args: Any, inner: NodeDef | None = None,
                 rebuild: Callable[[NodeDef], NodeDef] | None = None, **kwargs: Any):
        # inner names the single member on construction; _replace round-trips
        # it through `members` like every other composite
        if inner is None and 'members' in kwargs and 'inner' in kwargs['members']:
            inner = kwargs['members']['inner']
        self._user_rebuild = rebuild
        if inner is not None:
            kwargs.setdefault('members', {'inner': inner})
            if rebuild is not None:
                kwargs['rebuild'] = lambda new_m: rebuild(new_m['inner'])
        super().__init__(*args, **kwargs)

    def _replace(self, **changes: Any) -> NodeDef:
        base = dict(rebuild=self._user_rebuild)
        base.update(changes)
        return super()._replace(**base)

    @property
    def inner(self) -> NodeDef:
        return self.members['inner']

    @property
    def selectable(self) -> dict[str, NodeDef]:
        """What a name-based selection sees: the wrapped node's members, not
        this wrapper's own single slot."""
        return self.inner.selectable

    def map_leaves(self, fn: Callable[[NodeDef], Any]) -> Any:
        """Transparent pass-through: descend to inner leaf defs, skipping the
        wrapper layer so no 'inner' key pollutes the state metadata tree.
        Non-cyclic wrappers (e.g. scan internalizing state) return 0."""
        if not self.cyclic:                       # scan internalizes: no state left
            return 0
        return self.inner.map_leaves(fn)

    def map_state(self, state: Any, fn: Callable[[NodeDef, Any], Any]) -> Any:
        """Transparent pass-through: map fn over inner def and state subtrees."""
        if not self.cyclic:
            return state
        return self.inner.map_state(state, fn)


class Node:
    """A bound node: (def, param). THE pytree object.

    Leaves are the param; the def rides along as aux data. One class for all
    nodes, so treedefs of two bindings of the same def are equal.

    Views: cyclic nodes are called apply(state, input) -> (state, output);
    non-cyclic nodes apply(input) -> output (the empty state is supplied and
    discarded internally)."""

    __slots__ = ('ndef', 'param')

    ndef: NodeDef
    param: Param

    def __init__(self, ndef: NodeDef, param: Param):
        self.ndef = ndef
        self.param = param

    @property
    def name(self) -> str:
        return self.ndef.name

    @property
    def bound(self) -> bool:
        """A Node is the bound stage: (def, param)."""
        return True

    @property
    def cyclic(self) -> bool:
        return self.ndef.cyclic

    def parameterize(self, *args: Any, **kwargs: Any) -> Node:
        """Construct a fresh binding of this node's def (re-parameterize)."""
        return self.ndef.parameterize(*args, **kwargs)

    def init(self, state_input: Any = None, /, *, input: Any = None, **fields) -> State:
        """Construct this node's initial state: ONE `state_input` Struct (or loose
        fields packed into one), plus optionally a real value of the node's
        INPUT — its own channel, typed by apply_input_spec, never a state_input
        field. The value resolves/validates the def's spec on the way in."""
        if state_input is not None and fields:
            raise TypeError('pass ONE state_input Struct or loose fields, not both')
        bundle = state_input if state_input is not None else _as_bundle(fields)
        nd = self.ndef if input is None else _resolve(self.ndef, input)
        return nd.build_state(self.param, bundle, input=input)

    def with_input(self, input: Any) -> Node:
        """A copy of this node with its def's input spec bound to `input`."""
        return Node(self.ndef.with_input(input), self.param)

    def apply(self, *args: PyTree, **fields: Any) -> Output | tuple[State, Output]:
        """Cyclic: apply(state, input) -> (state, output).
        Non-cyclic: apply(input) -> output. ONE input pytree — or loose
        fields packed into one, the same boundary sugar as parameterize
        and init. State stays positional."""
        if self.ndef.cyclic:
            if fields:
                if len(args) != 1:
                    raise TypeError(f'{self.name} is cyclic: apply(state, input) or '
                                    f'apply(state, **fields); state stays positional')
                args = (args[0], _as_bundle(fields))
            if len(args) != 2:
                raise TypeError(f'{self.name} is cyclic: apply(state, input) -> (state, output)')
            state, input = args
            return self.ndef.apply_fn(self.param, state, input)
        if fields:
            if args:
                raise TypeError('pass ONE input pytree or loose fields, not both')
            args = (_as_bundle(fields),)
        if len(args) != 1:
            raise TypeError(f'{self.name} takes a single input pytree')
        # a non-cyclic node's state is empty, but not necessarily ():
        # a pipe of non-cyclic members carries a leafless Struct of ()s.
        # its own init constructs exactly that empty structure.
        _, output = self.ndef.apply_fn(self.param, self.ndef.build_state(self.param), args[0])
        return output

    def __call__(self, *args: PyTree, **fields: Any) -> Output | tuple[State, Output]:
        """Shorthand for apply."""
        return self.apply(*args, **fields)

    def scan(self, state: State | None, inputs: Input) -> tuple[State, Output]:
        """Run this cyclic node over a leading time axis via jax.lax.scan,
        in lax.scan's own order: state first, then the stream.

        state is required; None selects init() — the NEUTRAL start,
        explicitly. The default is only reachable when a neutral start
        exists: nodes whose starting values matter declare them as
        required init arguments (a trainer's model, a sensor's rng),
        so None fails loudly at exactly those nodes."""
        if not self.ndef.cyclic:
            raise TypeError(f'{self.name} is not cyclic; nothing to scan over')
        if state is None:
            state = self.init()
        return jax.lax.scan(self.apply, state, inputs)

    def __rshift__(self, other: GenericDef | NodeDef | Node) -> GenericDef | NodeDef | Node:
        """Serial composition; pipes stay flat: (a >> b) >> c == a >> b >> c."""
        from nodejax.compose import _compose
        return _compose(self, other)

    def __getattr__(self, name: str) -> Any:
        """'Param plays self', readable from outside too: def methods
        first (channel-bound: param and ndef injected; a state or rng
        channel is the caller's to pass by keyword, since a bare node
        holds no live state and no stream), then param-field forwarding
        — a bound node reads like the object its param tree describes,
        and the forwarding chains through nested nodes
        (stack.current_ctrl.motor.resistance). Real Node attributes
        (apply, init, param, name, ...) win over fields, and methods win
        over fields; node.param is always the unambiguous spelling."""
        methods = self.ndef.methods
        if methods and name in methods:
            return _bind_method(methods[name],
                                dict(param=lambda: self.param, ndef=lambda: self.ndef))
        param = object.__getattribute__(self, 'param')
        if isinstance(param, Struct) and name in param:
            return param[name]
        hints = []
        if methods:
            hints.append(f'methods: {sorted(methods)}')
        if isinstance(param, Struct):
            hints.append(f'param fields: {list(param.__keys__)}')
        hint = ('; ' + '; '.join(hints)) if hints else ''
        raise AttributeError(f'Node {self.name!r} has no attribute {name!r}{hint}')

    def __repr__(self) -> str:
        return f'Node({self.ndef!r}, param={self.param!r})'


def _node_flatten(node: Node) -> tuple[tuple[Param], NodeDef]:
    """Pytree flatten: param is the data, the def is aux."""
    return (node.param,), node.ndef


class _ParamHop:
    """Key entry for a Node's single child (its param). Renders as
    NOTHING in key paths: a node has exactly one data slot, so the hop
    carries no information — addresses read as the object graph reads,
    and keystr(path) IS the attribute chain
    ('.actuator.motor.resistance' <-> env.actuator.motor.resistance)."""
    __slots__ = ()

    def __str__(self):
        return ''

    def __repr__(self):
        return '<param>'


_PARAM_HOP = _ParamHop()


def _node_flatten_with_keys(node: Node):
    """Keyed flatten; the param hop is transparent in paths."""
    return ((_PARAM_HOP, node.param),), node.ndef


def _node_unflatten(ndef: NodeDef, children: tuple[Param]) -> Node:
    param, = children
    return Node(ndef, param)


jax.tree_util.register_pytree_with_keys(
    Node, _node_flatten_with_keys, _node_unflatten, _node_flatten)
