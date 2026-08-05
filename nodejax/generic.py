"""The static stage: blueprints awaiting static arguments.

A plain closure returning a NodeDef is a perfectly good generic node;
GenericDef exists for when statics should COMPOSE: a pipe of generics
exposes its members' statics as one nested tree, supplied at a single point
of use instead of threaded through constructors by hand. This is the same
Struct-over-members pattern the core applies to param and state, applied to
the static stage.
"""

from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable

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

    def specialize(self, *args: Any, **statics: Any) -> NodeDef | Node:
        """Bind static arguments; defaults merge under the supplied statics.

        AMBIENT STATICS: a key of the form '*.<name>' BROADCASTS — the
        value is delivered to every member, at any depth, whose generic
        DECLARES <name> (a named parameter of its closure); members that
        don't declare it are untouched, constants ignore it, and explicit
        member statics win over the broadcast. Threading mode-like flags
        through a construction graph dissolves into one entry at the
        single point of use:

            model_g.specialize(**{'*.train': False}, head={'train': True})
        """
        wilds = {k[2:]: statics.pop(k) for k in list(statics) if k.startswith('*.')}
        if wilds:
            statics = self._distribute(wilds, statics)
        return self.specialize_fn(*args, **_merge_statics(self.defaults, statics))

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

    def __call__(self, *args: Any, **statics: Any) -> NodeDef | Node:
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
