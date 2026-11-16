"""Single-member transparent `Wrapper` construction for transforms and decorators.

`Wrapper(inner=member)` declares one child Node whose parameter, state, method,
and nested-member surfaces pass straight through the wrapper without adding an
extra `Struct` nesting level. Calling `Wrapped(step, ...)` attaches authored
`(self, ...)` behavior, whereas `Wrapped.roles(...)` and `Wrapped(contract=...)`
override individual T3 roles or T4 `ContractCalls`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable

from nodejax.core.composite import _promote_members
from nodejax.core.contract import ContractCalls, _KEEP
from nodejax.core.definition import Captures, Def, Layout
from nodejax.frozendict import frozendict
from nodejax.core.generic import Generic
from nodejax.core.node import BaseNode, _is_node, _view
from nodejax.core.types import PyTree
from nodejax.struct import Struct


_EMPTY_MAPPING = frozendict()


def _resolved_member(definition: Def, name: str) -> Def:
    member = getattr(definition.members, name)
    shape = definition.contract.input_spec
    if shape is None or member.contract.input_spec is not None:
        return member
    return member.contract._resolve_def(shape, bundled=True)


def _transparent_calls(member_name: str, member: Def) -> ContractCalls:
    calls = member.calls

    if calls.param is not None:
        def param(definition, formed_input, rng):
            child = _resolved_member(definition, member_name)
            return child.contract.param(formed_input, rng)
        param_call = calls.param.copy(impl=param)
    else:
        param_call = None

    if calls.init is None:
        init_call = None
    elif calls.init.requires_input:
        def prime(definition, param, formed_input, input, rng):
            child = getattr(definition.members, member_name)
            return child.contract.prime(param, formed_input, input, rng)
        init_call = calls.init.copy(impl=prime)
    else:
        def init(definition, param, formed_input, rng):
            child = _resolved_member(definition, member_name)
            return child.contract.init(param, formed_input, rng)
        init_call = calls.init.copy(impl=init)

    def apply(definition, param, state, formed_input, rng):
        child = getattr(definition.members, member_name)
        return child.contract.apply(param, state, formed_input, rng)

    return ContractCalls(
        param=param_call,
        init=init_call,
        apply=calls.apply.copy(impl=apply),
    )


def _delegate_inherited_calls(member_name: str, member: Def,
                              calls: ContractCalls) -> ContractCalls:
    delegated = _transparent_calls(member_name, member)
    original = member.calls

    def role(given, source, through):
        if given is None:
            return None
        if source is not None and given.impl is source.impl:
            return given.copy(impl=through.impl)
        return given

    return ContractCalls(
        param=role(calls.param, original.param, delegated.param),
        init=role(calls.init, original.init, delegated.init),
        apply=role(calls.apply, original.apply, delegated.apply),
    )


def _transparent_def(member_name: str, member: Def, *, name: str,
                     captures: Captures,
                     tags: Iterable[str] | None = None,
                     boundaries: Mapping[str, Callable] = _EMPTY_MAPPING,
                     methods: Mapping[str, Callable] = _EMPTY_MAPPING,
                     destructurable: bool = True,
                     destructurable_state: bool = True,
                     externalized_param_paths: frozenset[str] = (
                         frozenset())) -> Def:
    """Builds a transparent `Def` that inherits `member`'s tags and delegates member methods."""
    members = Struct(**{member_name: member})

    def bind(replacements):
        return _transparent_def(
            member_name, getattr(replacements, member_name),
            name=name, captures=Captures(), tags=tags,
            boundaries=boundaries, methods=methods,
            destructurable=destructurable,
            destructurable_state=destructurable_state,
            externalized_param_paths=externalized_param_paths,
        )

    return Def(
        name=name,
        calls=_transparent_calls(member_name, getattr(members, member_name)),
        members=members,
        captures=captures,
        tags=member.tags if tags is None else frozenset(tags),
        boundaries=frozendict(boundaries),
        methods=frozendict(methods),
        tree=bind,
        layout=Layout(
            transparent_member=member_name,
            destructurable_param=destructurable,
            destructurable_state=destructurable_state,
            externalized_param_paths=externalized_param_paths,
        ),
    )


class Wrapped:
    """Holds the single named child Node declared via `Wrapper(name=member)`."""

    def __init__(self, member_name: str, operand: BaseNode | Generic) -> None:
        self.member_name = member_name
        self.operand = operand

    def roles(self, *, param=None, init=None, prime=None, apply=None,
              name: str | None = None,
              requires_input: bool | None = None,
              state_fields: tuple[str, ...] = (),
              param_takes_rng: bool | None = None,
              init_takes_rng: bool | None = None,
              apply_takes_rng: bool | None = None,
              input_spec: PyTree = _KEEP,
              apply_fields: tuple[str, ...] | None = None,
              open: bool = False,
              tags: Iterable[str] | None = None,
              boundary: Mapping[str, Callable] = _EMPTY_MAPPING,
              methods: Mapping[str, Callable] = _EMPTY_MAPPING,
              destructurable: bool = True,
              destructurable_state: bool = True,
              externalized_param_paths: frozenset[str] = frozenset()) -> BaseNode:
        """Builds a transparent wrapper by overriding selected T3 contract roles.

        Setting `requires_input` switches an inherited initializer between cold
        `init` and input-dependent `prime` when the matching replacement callable
        is supplied.
        """
        if not _is_node(self.operand):
            raise TypeError('Wrapper.roles requires a complete member Node')
        member_name = self.member_name
        operand = self.operand
        members, captures = _promote_members({member_name: operand})
        child = getattr(members, member_name)
        definition = _transparent_def(
            member_name, child,
            name=name or operand.name,
            captures=captures,
            tags=tags, boundaries=boundary, methods=methods,
            destructurable=destructurable,
            destructurable_state=destructurable_state,
            externalized_param_paths=externalized_param_paths,
        )
        calls = definition.contract._roles(
            param=param, init=init, prime=prime, apply=apply,
            requires_input=requires_input, state_fields=state_fields,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng,
            input_spec=input_spec,
            apply_fields=apply_fields, open=open,
        )
        options = dict(
            param=param, init=init, prime=prime, apply=apply,
            name=name, requires_input=requires_input, state_fields=state_fields,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng,
            input_spec=input_spec,
            apply_fields=apply_fields, open=open,
            tags=tags, boundary=boundary, methods=methods,
            destructurable=destructurable,
            destructurable_state=destructurable_state,
            externalized_param_paths=externalized_param_paths,
        )

        def bind(replacements):
            replacement = _view(getattr(replacements, member_name))
            return Wrapped(member_name, replacement).roles(**options)._def

        return _view(definition.copy(calls=calls, tree=bind))

    def __call__(self, apply: Callable | None = None, *, init=None,
                 name: str | None = None, contract: ContractCalls | None = None,
                 tags: Iterable[str] | None = None,
                 boundary: Mapping[str, Callable] = _EMPTY_MAPPING,
                 methods: Mapping[str, Callable] = _EMPTY_MAPPING,
                 destructurable: bool = True,
                 destructurable_state: bool = True,
                 rng_from: bool | None = None,
                 input_spec: PyTree = None) -> BaseNode | Generic:
        """Attaches authored `(self, ...)` behavior, metadata overrides, or `contract` calls."""
        if apply is not None:
            if (contract is not None or tags is not None
                    or boundary or methods):
                raise TypeError('authored Wrapper accepts behavior and name')
            from nodejax.core.compose import _wrap_build
            return _wrap_build(
                apply, self.operand, member=self.member_name,
                init=init, name=name, rng_from=rng_from,
                input_spec=input_spec)
        if init is not None or input_spec is not None:
            raise TypeError(
                'authored Wrapper init and input_spec require apply behavior')
        if rng_from is not None:
            raise TypeError('rng_from belongs to authored Wrapper behavior')
        if not _is_node(self.operand):
            raise TypeError('Wrapper requires a complete member Node')
        if contract is None:
            members, captures = _promote_members(
                {self.member_name: self.operand})
            definition = _transparent_def(
                self.member_name, getattr(members, self.member_name),
                name=name or self.operand.name,
                captures=captures,
                tags=tags, boundaries=boundary, methods=methods,
                destructurable=destructurable,
                destructurable_state=destructurable_state,
            )
        else:
            members, captures = _promote_members({self.member_name: self.operand})
            definition = Def(
                name=name or self.operand.name,
                calls=_delegate_inherited_calls(
                    self.member_name, getattr(members, self.member_name),
                    contract),
                members=members,
                captures=captures,
                tags=self.operand.tags if tags is None else frozenset(tags),
                boundaries=frozendict(boundary),
                methods=frozendict(methods),
                layout=Layout(
                    transparent_member=self.member_name,
                    destructurable_param=destructurable,
                    destructurable_state=destructurable_state,
                ),
            )
        return _view(definition)


class _WrapperDoor:
    """Declares a single transparent member (`Wrapper(inner=member)`) and returns `Wrapped`."""

    def __call__(self, *unnamed, **named: BaseNode | Generic) -> Wrapped:
        if unnamed or len(named) != 1:
            raise TypeError('Wrapper names exactly one member')
        (name, member), = named.items()
        return Wrapped(name, member)


Wrapper = _WrapperDoor()
