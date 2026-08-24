"""The complete low-level definition value.

`Def` is program information.  Public Nodes, bound Nodes, the T3 Contract
view, and authored scopes all project the same value; none owns a second copy
of its facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from nodejax.frozendict import frozendict
from nodejax.struct import Struct


@dataclass(eq=False, frozen=True)
class Captures:
    """Bound member values retained by a structural construction.

    ``Def.members`` always stores definitions.  When a ``PNode`` or
    ``PSNode`` is used as a member, its already-bound values cannot live in
    that definition tree: parameters and state are JAX data, not program
    structure.  ``Captures`` records those values beside the tree, keyed by
    the immediate member name.

    A capture becomes that member role's construction default.  Unless the
    caller replaces it, parent parameterization or initialization reuses the
    captured value instead of running the member constructor again.  Captured
    state that stores RNG keys is rekeyed when a fresh parent state is formed.
    Roles not present here remain open and are assembled normally from the
    parent's parameter and state inputs. The two role maps are immutable, but
    retain mapping semantics because their member-name keys are dynamic.

    Captures belong to the particular act of composition, not to the member
    ``Def`` itself.  Rebinding the definition tree clears them so a rewritten
    member starts from its own unbound construction stages.
    """

    param: frozendict = field(default_factory=frozendict)
    state: frozendict = field(default_factory=frozendict)

    def __post_init__(self):
        if type(self.param) is not frozendict or type(self.state) is not frozendict:
            raise TypeError('captured param and state must be frozendicts')


@dataclass(eq=False)
class Construction:
    """The canonical factory call and accumulated descendant specializations."""

    factory: Callable
    arguments: Struct
    named: frozendict = field(default_factory=frozendict)
    wildcards: frozendict = field(default_factory=frozendict)

    def copy(self, **changes) -> 'Construction':
        return replace(self, **changes)


@dataclass(eq=False)
class Layout:
    """How member-shaped definitions relate to their bound value trees."""

    kind: str = 'leaf'
    transparent_member: str | None = None
    param_members: frozenset[str] | None = None
    destructurable_param: bool = True
    destructurable_state: bool = True

    def __post_init__(self):
        if self.param_members is not None:
            self.param_members = frozenset(self.param_members)


@dataclass(eq=False)
class Def:
    """All information defining one NodeJAX computation.

    Equality and hashing use object identity. A Def is JAX static data and may
    close over arrays, for which structural dataclass equality is unsuitable.
    """

    name: str
    calls: Any
    members: Struct = field(default_factory=Struct)
    methods: frozendict = field(default_factory=frozendict)
    tags: frozenset[str] = frozenset()
    boundaries: frozendict = field(default_factory=frozendict)
    construction: Construction | None = None
    tree: Callable[[Struct], 'Def'] | None = None
    captures: Captures = field(default_factory=Captures)
    layout: Layout = field(default_factory=Layout)

    def __post_init__(self):
        from nodejax.contract import ContractCalls

        if type(self.calls) is not ContractCalls:
            raise TypeError(f'{self.name}: calls must be ContractCalls')
        if type(self.members) is not Struct:
            raise TypeError(f'{self.name}: members must be a Struct of Defs')
        if not all(type(member) is Def for member in self.members):
            raise TypeError(f'{self.name}: members must contain Def values')
        if type(self.methods) is not frozendict:
            raise TypeError(f'{self.name}: methods must be a frozendict')
        self.tags = frozenset(self.tags)
        if type(self.boundaries) is not frozendict:
            raise TypeError(f'{self.name}: boundaries must be a frozendict')
        if type(self.captures) is not Captures:
            raise TypeError(f'{self.name}: captures must be a Captures value')
        if type(self.layout) is not Layout:
            self.layout = Layout(**self.layout)

    def copy(self, **changes) -> 'Def':
        return replace(self, **changes)

    @property
    def parametric(self) -> bool:
        return self.calls.param is not None

    @property
    def cyclic(self) -> bool:
        return self.calls.init is not None

    @property
    def contract(self):
        from nodejax.contract import Contract
        return Contract(self)

    def bind_members(self, members: Struct) -> 'Def':
        """Re-enter the explicit tree-binding stage with named member Defs."""
        if type(members) is not Struct:
            raise TypeError('tree binding accepts a Struct of Def values')
        expected = set(self.members.__keys__)
        if set(members.__keys__) != expected:
            raise TypeError(
                f"cannot bind members of '{self.name}': expected "
                f'{sorted(expected)}, got {sorted(members)}')
        if self.tree is None:
            raise TypeError(
                f"cannot rewrite '{self.name}': its construction exposes no "
                'tree-binding stage')
        if not all(type(member) is Def for member in members):
            raise TypeError('tree binding accepts Def values, not Node views')

        built = self.tree(members)
        if type(built) is not Def:
            raise TypeError(
                f"the tree binder for '{self.name}' returned "
                f'{type(built).__name__}, not a Def')
        return built.copy(
            construction=self.construction,
            tree=self.tree,
            captures=Captures(),
        )
