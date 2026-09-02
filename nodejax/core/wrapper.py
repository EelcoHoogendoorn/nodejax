"""One-member structure-first construction."""

from __future__ import annotations

from collections.abc import Mapping

from nodejax.core.composite import _promote_members
from nodejax.core.contract import ContractCalls, _KEEP
from nodejax.core.definition import Captures, Def, Layout
from nodejax.frozendict import frozendict
from nodejax.core.node import _is_node, _view
from nodejax.struct import Struct


_EMPTY_MAPPING = frozendict()


def _resolved_member(definition: Def, name: str) -> Def:
    """Resolve only a member whose own input evidence is still unknown."""
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
    """Pass the child Def to unchanged calls and the wrapper Def to replacements."""
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
                     tags=None, boundaries: Mapping = _EMPTY_MAPPING,
                     methods: Mapping = _EMPTY_MAPPING,
                     destructurable=True, destructurable_state=True,
                     externalized_param_paths: frozenset[str] = (
                         frozenset())) -> Def:
    """A transparent wrapper's tags are the member's own unless replaced.
    It declares no methods of the member's: attribute lookup on a view or
    a member handle passes through to the member, which binds its own
    methods to the shared slots."""
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
    def __init__(self, member_name: str, operand):
        self.member_name = member_name
        self.operand = operand

    def roles(self, *, param=None, init=None, prime=None, apply=None,
              name=None, requires_input=None,
              param_takes_rng=None, init_takes_rng=None,
              apply_takes_rng=None, input_spec=_KEEP,
              apply_fields=None, open: bool = False,
              tags=None, boundary: Mapping = _EMPTY_MAPPING,
              methods: Mapping = _EMPTY_MAPPING,
              destructurable=True, destructurable_state=True,
              externalized_param_paths: frozenset[str] = frozenset()):
        """Build a wrapper by replacing roles with ordinary T3 functions.

        ``requires_input`` can promote an inherited init into a prime, or
        demote a prime into an init, when that replacement role is supplied.
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
            requires_input=requires_input,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng,
            input_spec=input_spec,
            apply_fields=apply_fields, open=open,
        )
        options = dict(
            param=param, init=init, prime=prime, apply=apply,
            name=name, requires_input=requires_input,
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

    def __call__(self, apply=None, *, init=None, name=None, contract=None,
                 tags=None, boundary: Mapping = _EMPTY_MAPPING,
                 methods: Mapping = _EMPTY_MAPPING,
                 destructurable=True, destructurable_state=True,
                 rng_from=None, input_spec=None):
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
    def __call__(self, *unnamed, **named):
        if unnamed or len(named) != 1:
            raise TypeError('Wrapper names exactly one member')
        (name, member), = named.items()
        return Wrapped(name, member)


Wrapper = _WrapperDoor()
