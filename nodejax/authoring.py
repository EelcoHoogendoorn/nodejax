"""The SUGAR layer: authored functions with human signatures transform into
the contract-shaped functions the NodeDef stores.

Authoring sugar is strictly a producer. Once lifted, NodeDefs are defined 100%
by their 3 contract functions (param_fn, init_fn, apply_fn). All downstream
composition, transforms, and tree rewrites lean exclusively on the 3-function
contract, independent of authoring signatures.

Public API:
  - node_def(apply, param=..., init=...): Define a NodeDef from authored functions.
  - derive(parent, ...): Functional extension/derivation of an existing NodeDef.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable, overload

from nodejax.struct import Struct
from nodejax.types import Param, State, ParamFn
from nodejax.core import (Node, NodeDef, _trivial_init_fn, _trivial_param_fn,
                                _bundle_spec_from_sig)
from nodejax.lifting import (
    KeyStream, _keys_only, _as_arrays, _no_var_params, _LeafStep, _with_leaf_step,
    _APPLY_LIFTS, _OBJECT_NAMES, _APPLY_RESERVED, _lift_apply_general, _apply_lift,
    _advance_rng, _lift_param, _lift_init, _init_requires_input, _state_spec_from_sig,
    _RESERVED_NAMES, _check_methods, _CHANNEL_RANK, _check_method_signature,
)


@overload
def node_def(apply: Callable, *, param: ParamFn | None = ...,
             init: Callable[..., State] | None = ..., name: str | None = ...,
             apply_input_spec: Any = ..., methods: dict[str, Callable] | None = ...,
             tags: tuple[str, ...] | frozenset[str] | None = ...,
             ) -> NodeDef | Node: ...
@overload
def node_def(apply: None = ..., *, param: ParamFn | None = ...,
             init: Callable[..., State] | None = ..., name: str | None = ...,
             apply_input_spec: Any = ..., methods: dict[str, Callable] | None = ...,
             tags: tuple[str, ...] | frozenset[str] | None = ...,
             ) -> Callable[[Callable], NodeDef | Node]: ...
def node_def(apply: Callable | None = None, *, param: ParamFn | None = None,
             init: Callable[..., State] | None = None, name: str | None = None,
             apply_input_spec: Any = None, methods: dict[str, Callable] | None = None,
             tags: tuple[str, ...] | frozenset[str] | None = None,
             ) -> NodeDef | Node | Callable[[Callable], NodeDef | Node]:
    """Author a node from functions with human signatures — the sugar frontend
    that computes the def's specs and transforms the fns into contract shape
    (see core).

    apply is required: one of the fixed whole-`input` signatures
    (input) | (param, input) | (state, input) | (param, state, input), or
    leading param/self and state followed by trailing INPUT FIELDS, which the
    sugar packs into a Struct bundle on the way in.
    """
    if apply is None:
        # decorator-with-arguments: @node_def(name=..., apply_input_spec=...)
        # evaluates node_def WITHOUT the apply first; the partial then
        # receives the decorated function as its `apply`
        return partial(node_def, param=param, init=init, name=name,
                       apply_input_spec=apply_input_spec, methods=methods, tags=tags)

    sig = tuple(inspect.signature(apply).parameters)
    _no_var_params(apply, sig)
    if param is not None:
        _no_var_params(param, inspect.signature(param).parameters)
    if init is not None:
        _no_var_params(init, inspect.signature(init).parameters)

    lift = _apply_lift(apply, sig)

    parametric = any(n in sig for n in _OBJECT_NAMES) or param is not None
    cyclic = 'state' in sig

    if parametric and param is None:
        raise TypeError(f"'{name or apply.__name__}' apply takes param, but no "
                        'param constructor was given; supply param= or drop '
                        'param from apply')
    if param is not None and not parametric:
        raise TypeError(f"'{name or apply.__name__}' was given a param= "
                        'constructor, but apply takes no param; name param/self '
                        'in apply or drop param=')
    if cyclic and init is None:
        raise TypeError(f"'{name or apply.__name__}' apply takes state, but no "
                        'init function was given; supply init= or drop state '
                        'from apply')
    if init is not None and not cyclic:
        raise TypeError(f"'{name or apply.__name__}' was given an init= "
                        'function, but apply takes no state; name state in apply '
                        'or drop init=')

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
        tags=frozenset(tags or ()),
    )
    return ndef if parametric else Node(ndef, ())


def derive(parent: NodeDef | Node, *, apply: Callable | None = None,
           init: Callable[..., State] | None = None, param: ParamFn | None = None,
           name: str | None = None, apply_input_spec: Any = None,
           state_input_spec: Any = None,
           methods: dict[str, Callable] | None = None,
           tags: tuple[str, ...] | frozenset[str] | None = None) -> NodeDef | Node:
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
        apply_fn = pd._apply_impl      # the unbound impl, not the bound accessor

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
        tags=pd.tags if tags is None else frozenset(tags),
    )
    return ndef if parametric else Node(ndef, ())
