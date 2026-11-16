"""Canonical low-level definition records for NodeJAX computations.

`Def` stores all static program facts for a Node in one immutable-by-convention
record. Public views (`Node`, `PNode`, `PSNode`), the T3 `Contract` surface,
and authored scopes all project the same `Def` without duplicating its
metadata, member tree, or executable calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable

from nodejax.frozendict import frozendict
from nodejax.struct import Struct

if TYPE_CHECKING:
    from nodejax.core.contract import Contract, ContractCalls


@dataclass(eq=False, frozen=True)
class Captures:
    """Stores pre-bound member parameters and states alongside `Def.members`.

    `Def.members` holds `Def` records only because parameters and state are JAX
    values rather than static program structure. When a `PNode` or `PSNode` is
    registered as a member, `Captures` retains its bound values keyed by immediate
    member name and supplies them as construction defaults unless the caller
    overrides that member. Captured states carrying PRNG keys are rekeyed whenever
    fresh parent state is initialized, and rebinding the member tree clears all
    captures.
    """

    param: frozendict = field(default_factory=frozendict)
    state: frozendict = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        if type(self.param) is not frozendict or type(self.state) is not frozendict:
            raise TypeError('captured param and state must be frozendicts')


@dataclass(eq=False)
class Construction:
    """Records a factory call and accumulated descendant specializations for replay.

    `arguments` holds public static arguments addressable by `specialize`, whereas
    `replay_arguments` holds extra positional or keyword values forwarded only when
    replaying the factory.
    """

    factory: Callable
    arguments: Struct
    named: frozendict = field(default_factory=frozendict)
    wildcards: frozendict = field(default_factory=frozendict)
    replay_arguments: Struct = field(default_factory=Struct)

    def copy(self, **changes) -> Construction:
        return replace(self, **changes)


@dataclass(eq=False)
class Layout:
    """Describes how a definition's member tree maps onto dense and sparse value trees."""

    kind: str = 'leaf'
    transparent_member: str | None = None
    param_members: frozenset[str] | None = None
    destructurable_param: bool = True
    destructurable_state: bool = True
    externalized_param_paths: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.param_members is not None:
            self.param_members = frozenset(self.param_members)
        self.externalized_param_paths = frozenset(self.externalized_param_paths)


@dataclass(eq=False)
class Def:
    """Holds the complete static definition of one NodeJAX computation.

    Equality and hashing use object identity so `Def` values act as JAX static
    data even when their closures retain array constants.
    """

    name: str
    calls: ContractCalls
    members: Struct = field(default_factory=Struct)
    methods: frozendict = field(default_factory=frozendict)
    tags: frozenset[str] = frozenset()
    boundaries: frozendict = field(default_factory=frozendict)
    construction: Construction | None = None
    tree: Callable[[Struct], Def] | None = None
    captures: Captures = field(default_factory=Captures)
    layout: Layout = field(default_factory=Layout)

    def __post_init__(self) -> None:
        from nodejax.core.contract import ContractCalls

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

    def copy(self, **changes) -> Def:
        return replace(self, **changes)

    @property
    def parametric(self) -> bool:
        return self.calls.param is not None

    @property
    def cyclic(self) -> bool:
        return self.calls.init is not None

    @property
    def contract(self) -> Contract:
        from nodejax.core.contract import Contract
        return Contract(self)

    def bind_members(self, members: Struct) -> Def:
        """Rebuilds the definition with a replacement `Struct` of member `Def` values."""
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
