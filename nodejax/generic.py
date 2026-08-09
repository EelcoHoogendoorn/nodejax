"""The static stage: blueprints awaiting static arguments.

Plain Python closures do not compose at the static stage. GenericDef wraps
static blueprints so static trees compose as nested Structs, supplied at a single
point of use instead of threaded through constructors by hand. This is the same
Struct-over-members pattern the core applies to param and state, applied to the
static stage.
"""

from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable

from nodejax.struct import Struct
from nodejax.types import StaticTree
from nodejax.core import Node, NodeDef


class GenericDef:
    """A blueprint awaiting static arguments — the binding stage before NodeDef.

    Statics with two natures coexist: DERIVED statics stay ordinary closure
    logic inside specialize_fn (an author wiring hidden sizes together);
    FREE statics surface through composition. defaults merge under the
    supplied statics, so authored generics can pre-configure members.

    Transforms commute with specialization (see _over_generic):
    transform(G).specialize(s) == transform(G.specialize(s)).
    """

    name: str

    @property
    def bound(self) -> bool:
        """A generic is the stage BEFORE binding: never bound."""
        return False

    specialize_fn: Callable[..., NodeDef | Node]
    defaults: StaticTree
    members: dict[str, GenericDef] | None

    def __init__(self, name: str, specialize_fn: Callable[..., NodeDef | Node],
                 defaults: StaticTree | None = None,
                 members: dict[str, GenericDef] | None = None):
        self.name = name
        self.specialize_fn = specialize_fn
        self.defaults = defaults or {}
        self.members = members

    @property
    def static_input_spec(self) -> Struct:
        """The member-keyed tree of static parameters expected by this generic (REQUIRED or default)."""
        if self.members is not None:
            return Struct(**{nm: d.static_input_spec for nm, d in self.members.items()})
        sig = inspect.signature(self.specialize_fn)
        spec_fields = {}
        for nm, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                if nm in self.defaults:
                    spec_fields[nm] = self.defaults[nm]
                elif p.default is not inspect.Parameter.empty:
                    spec_fields[nm] = p.default
                else:
                    from nodejax.core import REQUIRED
                    spec_fields[nm] = REQUIRED
        return Struct(**spec_fields)

    def specialize(self, *args: Any, **statics: Any) -> GenericDef | NodeDef | Node:
        """Bind static arguments; defaults merge under the supplied statics.

        Supports dot-notation pathing (e.g. `linear.in_features=4`), wildcard
        broadcasting (`*.train=False`), and partial binding (returns a refined
        GenericDef if required statics remain unfulfilled).
        """
        statics = _unflatten_dot_paths(statics)
        wilds = {k[2:]: statics.pop(k) for k in list(statics) if k.startswith('*.')}
        if wilds:
            statics = self._distribute(wilds, statics)
        merged = _merge_statics(self.defaults, statics)
        try:
            return self.specialize_fn(*args, **merged)
        except TypeError as e:
            # If arguments are partially supplied, return a refined GenericDef carrying merged defaults
            if self.members is None:
                sig = inspect.signature(self.specialize_fn)
                try:
                    sig.bind_partial(*args, **merged)
                    return GenericDef(self.name, self.specialize_fn, defaults=merged, members=self.members)
                except TypeError:
                    pass
            raise e

    def _distribute(self, wilds: StaticTree, statics: StaticTree) -> StaticTree:
        """Resolve broadcast statics one level: composites re-inject the
        broadcast into every member subtree (deeper recursion happens in
        the members' own specialize); declared-signature leaves take what
        they declare, under explicit statics; opaque forwarders (**kwargs,
        e.g. deferred transforms) pass the broadcast through unresolved."""
        if self.members is not None:
            out = dict(statics)
            for name in self.members:
                sub = out.get(name, {})
                if isinstance(sub, dict):
                    out[name] = {**{f'*.{k}': v for k, v in wilds.items()}, **sub}
            return out
        params = inspect.signature(self.specialize_fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return {**{f'*.{k}': v for k, v in wilds.items()}, **statics}
        accepted = {k: v for k, v in wilds.items() if k in params}
        return _merge_statics(accepted, statics)

    def __call__(self, *args: Any, **statics: Any) -> GenericDef | NodeDef | Node:
        """Shorthand for specialize."""
        return self.specialize(*args, **statics)

    def __rshift__(self, other: GenericDef | NodeDef | Node) -> GenericDef:
        """Serial composition; member statics compose as a nested tree."""
        from nodejax.compose import _compose
        return _compose(self, other)

    # --- guard rails ---
    def parameterize(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f'{self.name} is a GenericDef; specialize it before parameterizing')

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f'{self.name} is a GenericDef; specialize it before applying')

    def __repr__(self) -> str:
        return f'GenericDef({self.name})'


def _unflatten_dot_paths(statics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in statics.items():
        if '.' in key and not key.startswith('*.'):
            parts = key.split('.')
            curr = out
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = value
        else:
            if isinstance(value, dict) and key in out and isinstance(out[key], dict):
                out[key] = _merge_statics(out[key], value)
            else:
                out[key] = value
    return out


def generic(fn: Callable[..., NodeDef | Node] | None = None, *,
            name: str | None = None, **defaults: Any,
            ) -> GenericDef | Callable[[Callable], GenericDef]:
    """Lift a def-returning closure into a composable GenericDef.
    Usable bare (@generic) or with defaults (@generic(hidden=32))."""
    if fn is None:
        return partial(generic, name=name, **defaults)
    return GenericDef(name or fn.__name__, fn, defaults=defaults)


def _merge_statics(defaults: StaticTree, overrides: StaticTree) -> StaticTree:
    """Nested dict merge; overrides win leaf-wise."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_statics(merged[key], value)
        else:
            merged[key] = value
    return merged


def _over_generic(transform: Callable) -> Callable:
    """Lift a transform over the static stage: applied to a GenericDef it
    defers, so that transform(G).specialize(s) == transform(G.specialize(s))."""
    @wraps(transform)
    def wrapper(node, *args, **kwargs):
        if isinstance(node, GenericDef):
            return GenericDef(
                f'{transform.__name__}({node.name})',
                lambda *s_args, **statics: transform(
                    node.specialize(*s_args, **statics), *args, **kwargs),
            )
        return transform(node, *args, **kwargs)
    return wrapper
