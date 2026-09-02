"""Structure-first construction of ordinary definitions."""

from __future__ import annotations

from collections.abc import Mapping

from nodejax.core.contract import (
    ApplyCall, CallField, CallForm, Contract, ContractCalls, InitCall, ParamCall,
    _lower_apply, _lower_init, _lower_param,
)
from nodejax.core.definition import Captures, Def, Layout
from nodejax.frozendict import frozendict
from nodejax.core.generic import Generic, is_generic
from nodejax.core.lifting import _check_methods
from nodejax.core.node import Node, _is_node, _view
from nodejax.struct import Struct


_DEFAULT = object()
_EMPTY_MAPPING = frozendict()


def _construction_form(value: Struct, role: str) -> CallForm:
    if type(value) is not Struct:
        raise TypeError(f'{role} must be a Struct declaration')
    return CallForm.from_values(value)


def _member_roles(definitions: Struct, captures: Captures, apply, *,
                  param=None, init=None, prime=None, name=None,
                  param_form: CallForm, state_form: CallForm,
                  apply_fields=None, input_contract=None,
                  input_spec=_DEFAULT, open: bool = False,
                  requires_input=None,
                  param_takes_rng=None, init_takes_rng=None,
                  apply_takes_rng=None, tags=(),
                  methods: Mapping = _EMPTY_MAPPING) -> Def:
    """Lower bundle-based role functions for one named member structure."""
    from nodejax.core.binding import REQUIRED

    if apply is None:
        raise TypeError('Composite.roles requires apply=')
    if input_contract is not None and type(input_contract) is not Contract:
        raise TypeError('input_contract must be a Contract')
    if input_contract is not None and apply_fields is not None:
        raise TypeError('choose input_contract= or apply_fields=')

    if requires_input is None:
        requires_input = any(
            member.cyclic
            and name not in captures.state
            and member.contract.init_requires_input
            for name, member in definitions.__items__)
    if type(requires_input) is not bool:
        raise TypeError('requires_input must be a bool or None')

    if param_takes_rng is None:
        param_takes_rng = any(
            member.parametric
            and name not in captures.param
            and member.contract.param_takes_rng
            for name, member in definitions.__items__)
    if init_takes_rng is None:
        from nodejax.core.binding import _has_rng_deep
        init_takes_rng = any(
            member.cyclic
            and ((_has_rng_deep(captures.state[name])
                  if name in captures.state else
                  member.contract.init_takes_rng))
            for name, member in definitions.__items__)
    if apply_takes_rng is None:
        apply_takes_rng = any(
            member.contract.apply_takes_rng for member in definitions)
    for role, value in (
            ('param', param_takes_rng),
            ('init', init_takes_rng),
            ('apply', apply_takes_rng)):
        if type(value) is not bool:
            raise TypeError(f'{role}_takes_rng must be a bool or None')

    noop = lambda *_: None
    param_call = None
    if param is not None:
        param_call = _lower_param(
            param, ParamCall(noop, param_form, param_takes_rng))

    initializer = prime if requires_input else init
    init_call = None
    if initializer is not None:
        init_call = _lower_init(
            initializer,
            InitCall(
                noop, state_form, init_takes_rng,
                requires_input),
            primes=requires_input,
        )
    elif any(member.cyclic for member in definitions):
        needed = 'prime=' if requires_input else 'init='
        raise TypeError(f'cyclic Composite.roles requires {needed}')

    if input_contract is not None:
        apply_form = input_contract._apply_form
        inherited_spec = input_contract.input_spec
    else:
        if apply_fields is None:
            raise TypeError(
                'Composite.roles requires apply_fields= or input_contract=')
        apply_form = CallForm.from_values(
            Struct(**{field: REQUIRED for field in apply_fields}), open=open)
        inherited_spec = None
    evidence = inherited_spec if input_spec is _DEFAULT else input_spec
    apply_call = _lower_apply(
        apply,
        ApplyCall(noop, apply_form, evidence, apply_takes_rng),
    )
    return Def(
        name=name or 'composite',
        calls=ContractCalls(
            param=param_call, init=init_call, apply=apply_call),
        members=definitions,
        captures=captures,
        tags=frozenset(tags),
        methods=_check_methods(methods),
        layout=Layout(kind='composite'),
    )


def _promote_members(members: Mapping[str, object]):
    """Split construction views into member definitions and explicit captures."""
    for name, member in members.items():
        if not _is_node(member):
            raise TypeError(
                f'member {name!r} is {type(member).__name__}, not a Node')

    bindings = {
        name: dict(member._binding_items())
        for name, member in members.items()
    }
    return (
        Struct(**{name: member._def for name, member in members.items()}),
        Captures(
            param=frozendict({
                name: binding['param']
                for name, binding in bindings.items()
                if 'param' in binding
            }),
            state=frozendict({
                name: binding['state']
                for name, binding in bindings.items()
                if 'state' in binding
            }),
        ),
    )


class Members(Mapping):
    """Named construction values waiting for behavior."""

    def __init__(self, members: dict):
        self._members = Struct(**members)

    def __getitem__(self, name):
        return self._members[name]

    def __iter__(self):
        return iter(self._members.__keys__)

    def __len__(self):
        return len(self._members)

    def __getattr__(self, name):
        if name in self._members:
            return self._members[name]
        raise AttributeError(name)

    def roles(self, apply, *, param=None, init=None, prime=None, name=None,
              param_input=_DEFAULT, state_input=_DEFAULT,
              apply_fields=None, input_contract=None,
              input_spec=_DEFAULT, open: bool = False,
              requires_input=None,
              param_takes_rng=None, init_takes_rng=None,
              apply_takes_rng=None, tags=()):
        """Build a composite from ordinary bundle-based role functions."""
        members = dict(self._members.__items__)
        options = dict(
            param=param, init=init, prime=prime, name=name,
            param_input=param_input, state_input=state_input,
            apply_fields=apply_fields, input_contract=input_contract,
            input_spec=input_spec, open=open,
            requires_input=requires_input,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng, tags=tags,
        )
        if any(is_generic(member) for member in members.values()):
            return Generic(
                name or 'composite',
                lambda **resolved: Members(resolved).roles(apply, **options),
                Struct(**members),
            )

        param_form = (_DEFAULT if param_input is _DEFAULT else
                      _construction_form(param_input, 'param_input'))
        state_form = (_DEFAULT if state_input is _DEFAULT else
                      _construction_form(state_input, 'state_input'))
        return self._roles_with_forms(
            apply,
            param=param, init=init, prime=prime, name=name,
            param_form=param_form, state_form=state_form,
            apply_fields=apply_fields, input_contract=input_contract,
            input_spec=input_spec, open=open,
            requires_input=requires_input,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng, tags=tags,
        )

    def _roles_with_forms(
            self, apply, *, param=None, init=None, prime=None, name=None,
            param_form=_DEFAULT, state_form=_DEFAULT,
            apply_fields=None, input_contract=None,
            input_spec=_DEFAULT, open: bool = False,
            requires_input=None,
            param_takes_rng=None, init_takes_rng=None,
            apply_takes_rng=None, tags=(),
            methods: Mapping = _EMPTY_MAPPING):
        """Build roles from canonical construction forms."""
        members = dict(self._members.__items__)
        role_options = dict(
            param=param, init=init, prime=prime, name=name,
            apply_fields=apply_fields, input_contract=input_contract,
            input_spec=input_spec, open=open,
            requires_input=requires_input,
            param_takes_rng=param_takes_rng,
            init_takes_rng=init_takes_rng,
            apply_takes_rng=apply_takes_rng, tags=tags,
            methods=methods,
        )
        if any(is_generic(member) for member in members.values()):
            return Generic(
                name or 'composite',
                lambda **resolved: Members(resolved)._roles_with_forms(
                    apply, param_form=param_form, state_form=state_form,
                    **role_options),
                Struct(**members),
            )

        definitions, captures = _promote_members(members)

        def forms(current: Struct, bound: Captures) -> tuple[CallForm, CallForm]:
            return (
                member_param_input(current, bound)
                if param_form is _DEFAULT else param_form,
                member_state_input(current, bound)
                if state_form is _DEFAULT else state_form,
            )

        current_param_form, current_state_form = forms(
            definitions, captures)
        definition = _member_roles(
            definitions, captures, apply,
            param_form=current_param_form,
            state_form=current_state_form,
            **role_options)

        def bind(replacements):
            replacement_param_form, replacement_state_form = forms(
                replacements, Captures())
            return _member_roles(
                replacements, Captures(), apply,
                param_form=replacement_param_form,
                state_form=replacement_state_form,
                **role_options)

        return _view(definition.copy(tree=bind))

    def __call__(self, apply=None, *, param=None, init=None,
                 apply_input_spec=None, name=None,
                 methods: Mapping = _EMPTY_MAPPING,
                 contract: ContractCalls | None = None):
        members = dict(self._members.__items__)
        methods = frozendict(methods)
        if any(is_generic(member) for member in members.values()):
            if contract is not None:
                raise TypeError('a contract requires complete member definitions')

            def build(**resolved):
                return Members(resolved)(
                    apply, param=param, init=init,
                    apply_input_spec=apply_input_spec, name=name,
                    methods=methods)

            return Generic(name or 'composite', build, Struct(**members))

        if apply is not None:
            if contract is not None:
                raise TypeError('choose authored behavior or contract calls')
            from nodejax.core.compose import composite
            return composite(
                apply, members=members, param=param, init=init,
                apply_input_spec=apply_input_spec, name=name,
                methods=methods)

        if any(value is not None
               for value in (param, init, apply_input_spec)):
            raise TypeError('low-level Composite accepts contract= and name=')
        if methods:
            raise TypeError('methods require authored Composite behavior')
        if contract is None:
            raise TypeError('supply authored apply or contract=')
        definitions, captures = _promote_members(members)
        return _view(Def(
            name=name or 'composite',
            calls=contract,
            members=definitions,
            captures=captures,
        ))


class _CompositeDoor:
    def __call__(self, **members) -> Members:
        return Members(members)


Composite = _CompositeDoor()


def member_param_input(members: Struct, captures: Captures) -> CallForm:
    empty = CallForm.from_values(Struct())
    return CallForm(Struct(**{
        name: (CallField.complete(captures.param[name])
               if name in captures.param else
               CallField.nested(member.calls.param.form)
               if member.parametric else CallField.nested(empty))
        for name, member in members.__items__
    }))


def member_state_input(members: Struct, captures: Captures) -> CallForm:
    empty = CallForm.from_values(Struct())
    return CallForm(Struct(**{
        name: (CallField.complete(captures.state[name])
               if name in captures.state else
               CallField.nested(member.calls.init.form)
               if member.cyclic else CallField.nested(empty))
        for name, member in members.__items__
    }))
