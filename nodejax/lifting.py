"""Internal signature lifting engine for authored NodeJax functions.

Handles signature inspection, parameter/state lifting, and channel injection
for leaf nodes and authored functions.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct, Aux
from nodejax.types import Param, State, ApplyFn
from nodejax.core import REQUIRED, _has_rng, _METHOD_CHANNELS, split_aux


class KeyStream:
    """Locally-mutable key dispenser: next() yields a fresh key per call.
    The mutation is local to the constructor's scope — it never escapes —
    so referential transparency holds. Param constructors declaring rng
    receive their offered key pre-wrapped."""

    def __init__(self, key: Any):
        self._key = key

    def next(self) -> Any:
        self._key, sub = jax.random.split(self._key)
        return sub

    @property
    def shape(self) -> tuple[int, ...]:
        return getattr(self._key, 'shape', jnp.asarray(self._key).shape)

    @property
    def dtype(self) -> Any:
        return getattr(self._key, 'dtype', jnp.asarray(self._key).dtype)

    @property
    def ndim(self) -> int:
        return getattr(self._key, 'ndim', jnp.asarray(self._key).ndim)

    def __jax_array__(self) -> Any:
        return self.next()


def _keystream_flatten(ks: KeyStream):
    return (ks._key,), None


def _keystream_unflatten(aux: None, children: tuple[Any, ...]) -> KeyStream:
    return KeyStream(children[0])


jax.tree_util.register_pytree_node(KeyStream, _keystream_flatten, _keystream_unflatten)


def _keys_only(tree: Any) -> Any:
    """Collapse any KeyStream leaf to a fresh raw key at lift exit."""
    return jax.tree.map(lambda leaf: leaf.next() if type(leaf) is KeyStream else leaf,
                        tree, is_leaf=lambda x: type(x) is KeyStream)


def _as_arrays(tree: Any) -> Any:
    """Cast every leaf of a constructed tree to a JAX array."""
    return jax.tree.map(jnp.asarray, tree)


def _no_var_params(fn: Callable, role: str) -> dict:
    """Refuse *args/**kwargs in authored signatures."""
    params = inspect.signature(fn).parameters
    if any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
           for p in params.values()):
        raise TypeError(f'{role} takes a closed set of named parameters; *args/**kwargs '
                        'have no meaning against a bundle')
    return params


class _LeafStep:
    """Transient step object passed as `self` to authored leaf functions."""
    __slots__ = ('_nd', 'param', 'state', '_aux')

    def __init__(self, ndef: Any, param: Any, state: Any):
        self._nd = ndef
        self.param = param
        self.state = state
        self._aux = {}

    def sow(self, **kwargs: Any) -> None:
        """Sow auxiliary metrics, losses, or taps into self._aux."""
        self._aux.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        if self.param is not None and name in self.param:
            return self.param[name]
        if self.state is not None and name in self.state:
            val = self.state[name]
            if name == 'rng' and type(val) is not KeyStream:
                return KeyStream(val)
            return val
        methods = self._nd.methods if self._nd is not None else None
        if methods and name in methods:
            from nodejax.core import _bind_method
            return _bind_method(methods[name],
                                dict(param=lambda: self.param,
                                     state=lambda: self.state,
                                     ndef=lambda: self._nd))
        return getattr(self._nd, name)


def _with_leaf_step(fn: Callable) -> ApplyFn:
    """Lift an apply function declaring `self` on a leaf node."""
    def wrapped(p: Any, s: Any, i: Any) -> Any:
        nd = getattr(p, 'ndef', None)
        step_self = _LeafStep(nd, p, s)
        res = fn(step_self, s, i)
        if step_self._aux:
            state, raw_out = res if type(res) is tuple and len(res) == 2 else (s, res)
            clean_out, direct_aux = split_aux(raw_out)
            aux_fields = {}
            if direct_aux is not None:
                if type(direct_aux) is Struct:
                    for k in direct_aux.__keys__:
                        aux_fields[k] = direct_aux[k]
                elif type(direct_aux) is dict:
                    aux_fields.update(direct_aux)
            aux_fields.update(step_self._aux)
            return state, (clean_out, Aux(**aux_fields))
        return res
    return wrapped


_APPLY_LIFTS: dict[tuple[str, ...], Callable[[Callable], ApplyFn]] = {
    ('input',):                          lambda f: lambda p, s, i: (s, f(i)),
    ('param', 'input'):                  lambda f: lambda p, s, i: (s, f(p, i)),
    ('self', 'input'):                   lambda f: _with_leaf_step(lambda self_obj, s, i: (s, f(self_obj, i))),
    ('state', 'input'):                  lambda f: lambda p, s, i: f(s, i),
    ('param', 'state', 'input'):         lambda f: f,
    ('self', 'state', 'input'):          lambda f: _with_leaf_step(lambda self_obj, s, i: f(self_obj, s, i)),
    ('ndef', 'input'):                  lambda f: lambda p, s, i: (s, f(getattr(p, 'ndef', p), i)),
    ('ndef', 'param', 'input'):         lambda f: lambda p, s, i: (s, f(getattr(p, 'ndef', p), p, i)),
    ('ndef', 'state', 'input'):         lambda f: lambda p, s, i: f(getattr(p, 'ndef', p), s, i),
    ('ndef', 'param', 'state', 'input'): lambda f: lambda p, s, i: f(getattr(p, 'ndef', p), p, s, i),
}

_OBJECT_NAMES = ('param', 'self')
_APPLY_RESERVED = frozenset(_OBJECT_NAMES) | {'state', 'ndef'}


def _lift_apply_general(apply: Callable) -> ApplyFn:
    """Lift an apply function whose trailing parameters are unpacked input fields."""
    params = list(inspect.signature(apply).parameters)
    obj = next((p for p in params if p in _OBJECT_NAMES), None)
    has_state = 'state' in params
    has_ndef = 'ndef' in params
    fields = [p for p in params if p not in _APPLY_RESERVED]

    def apply_fn(param: Any, state: Any, input: Any) -> Any:
        kw: dict = {}
        step_self = _LeafStep(getattr(param, 'ndef', None), param, state) if obj == 'self' else None
        if has_ndef:
            kw['ndef'] = getattr(param, 'ndef', None)
        if obj == 'self':
            kw['self'] = step_self
        elif obj is not None:
            kw[obj] = param
        if has_state:
            kw['state'] = state
        for f in fields:
            v = input[f]
            kw[f] = KeyStream(v) if f == 'rng' and type(v) is not KeyStream else v
        raw_out = _keys_only(apply(**kw))
        if step_self is not None and step_self._aux:
            clean_out, direct_aux = split_aux(raw_out)
            aux_fields = {}
            if direct_aux is not None:
                if type(direct_aux) is Struct:
                    for k in direct_aux.__keys__:
                        aux_fields[k] = direct_aux[k]
                elif type(direct_aux) is dict:
                    aux_fields.update(direct_aux)
            aux_fields.update(step_self._aux)
            out = (clean_out, Aux(**aux_fields))
        else:
            out = raw_out
        return out if has_state else (state, out)

    return apply_fn


def _apply_lift(apply: Callable, sig: tuple) -> ApplyFn:
    """Select the apply lift: a fixed whole-`input` pattern, else the general unpack."""
    lift = _APPLY_LIFTS.get(sig)
    if lift is not None:
        return lift
    if 'input' in sig or not any(p not in _APPLY_RESERVED for p in sig):
        raise TypeError(f'apply must be a reserved signature {list(_APPLY_LIFTS)} or '
                        f'leading param/state then input FIELDS, got {sig}')
    return _lift_apply_general


def _advance_rng(apply_fn: ApplyFn) -> ApplyFn:
    """Auto-advance the reserved 'rng' state field."""
    def wrapped(p, s, i):
        if _has_rng(s):
            next_key, use_key = jax.random.split(s.rng)
            new_s, out = apply_fn(p, s.replace(rng=use_key), i)
            return new_s.replace(rng=next_key), out
        return apply_fn(p, s, i)
    return wrapped


def _lift_param(ctor: Callable) -> Callable:
    """Transform an authored param constructor into param_fn(ndef, param_input)."""
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


def _lift_init(init: Callable[..., State]) -> Callable[..., State]:
    """Lift an authored init function into init_fn(ndef, param, state_input, input=None)."""
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
                kw[nm] = input
            elif nm == 'rng':
                if 'rng' in state_input:
                    kw[nm] = KeyStream(state_input.rng)
            elif nm == 'state':
                kw[nm] = state_input
            elif nm in state_input:
                kw[nm] = state_input[nm]
        return _keys_only(init(**kw))

    return init_fn


def _init_requires_input(init: Callable) -> bool:
    """Check if an init function requires input."""
    params = inspect.signature(init).parameters
    return 'input' in params and params['input'].default is inspect.Parameter.empty


def _state_spec_from_sig(init: Callable) -> Struct:
    """Compute state_input bundle spec from init signature."""
    params = inspect.signature(init).parameters
    if 'rng' in params and params['rng'].default is not inspect.Parameter.empty:
        raise TypeError(f'{init.__name__}: rng never has a default; a '
                        'stochastic init requires its key')
    return Struct(**{
        nm: (REQUIRED if p.default is inspect.Parameter.empty else p.default)
        for nm, p in params.items()
        if nm not in ('param', 'self', 'ndef', 'input', 'state')
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)})


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


_CHANNEL_RANK = {'ndef': 0, 'param': 1, 'state': 2, 'rng': 3}


def _check_method_signature(nm: str, fn: Callable) -> None:
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
