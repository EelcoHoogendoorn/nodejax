"""Canonical T4 calls and the T3 view over them.

Authoring syntax is lowered into :class:`ContractCalls`. Those records are
definition data: fixed framework calls plus the facts needed to invoke them.
The :class:`Contract` view is the narrower surface used by transforms and
composition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any, Callable

from nodejax.frozendict import frozendict
from nodejax.rng import MaybeKeyStream
from nodejax.struct import Struct
from nodejax.types import PyTree


_KEEP = object()


@dataclass(eq=False, frozen=True)
class CallField:
    """One field in a canonical T4 call form.

    A complete field carries one value wholesale, even when that value is a
    ``Struct`` or ``None``. A nested field carries another ``CallForm`` whose
    fields are constructed independently. The explicit variant is the
    semantic fact; the Python type of ``content`` never selects it.
    """

    is_nested: bool
    content: Any

    def __post_init__(self):
        if type(self.is_nested) is not bool:
            raise TypeError('CallField.is_nested must be a bool')
        if self.is_nested and type(self.content) is not CallForm:
            raise TypeError('a nested CallField must contain a CallForm')

    @classmethod
    def complete(cls, default: Any) -> 'CallField':
        """Declare one complete value with ``default`` as its fallback."""
        return cls(False, default)

    @classmethod
    def nested(cls, form: 'CallForm') -> 'CallField':
        """Declare a recursively constructed field described by ``form``."""
        return cls(True, form)

    @property
    def declaration(self) -> Any:
        """Project this field into the lossy public Struct declaration."""
        return self.content.declaration if self.is_nested else self.content


@dataclass(eq=False)
class CallForm:
    """Canonical T4 structure for one role's ordered call fields.

    ``fields`` records complete-versus-nested semantics explicitly. ``open``
    permits additional complete fields while retaining validation and default
    formation for the declared fields. Public contract specs are projections
    of this form and are never used to reconstruct it.
    """

    fields: Struct
    open: bool = False

    def __post_init__(self):
        if type(self.fields) is not Struct:
            raise TypeError('CallForm.fields must be a Struct')
        if not all(type(field) is CallField for field in self.fields):
            raise TypeError('CallForm.fields must contain CallField values')
        if type(self.open) is not bool:
            raise TypeError('CallForm.open must be a bool')

    @classmethod
    def from_values(cls, declaration: Struct, *,
                    open: bool = False) -> 'CallForm':
        """Treat every field in a public declaration as one complete value."""
        if type(declaration) is not Struct:
            raise TypeError('CallForm declaration must be a Struct')
        return cls(Struct(**{
            name: CallField.complete(default)
            for name, default in declaration.__items__
        }), open=open)

    @property
    def declaration(self) -> Struct:
        """Project the canonical form to its public Struct declaration."""
        return Struct(**{
            name: field.declaration for name, field in self.fields.__items__
        })

    def without(self, *names: str) -> 'CallForm':
        """Return the same form without the named top-level fields."""
        return type(self)(self.fields.without(*names), open=self.open)

    def feed(self, value: Any) -> Struct:
        """Form one computed wire for this call."""
        if self.open:
            return self.feed_bundle(value)
        names = self.fields.__keys__
        if not names:
            raise TypeError('a zero-input call cannot consume a wire')
        if len(names) == 1:
            return Struct(**{names[0]: value})
        if not issubclass(type(value), Struct):
            raise TypeError('a multi-input call requires a Struct bundle')
        return value

    def feed_bundle(self, bundle: Struct) -> Struct:
        """Accept one explicitly formed call bundle without adaptation."""
        if not issubclass(type(bundle), Struct):
            raise TypeError('a formed call bundle must be a Struct')
        return bundle

    def intake(self, bundle: Struct) -> Any:
        """Recover the wire represented by one explicitly formed bundle."""
        if not issubclass(type(bundle), Struct):
            raise TypeError('a formed call bundle must be a Struct')
        names = self.fields.__keys__
        if not self.open and len(names) == 1:
            return bundle[names[0]]
        return bundle


@dataclass(eq=False)
class ParamCall:
    """``impl(definition, formed_input, rng) -> param``."""

    impl: Callable
    form: CallForm
    takes_rng: bool = False
    reads_def: bool = False

    def __post_init__(self):
        _check_call(self.impl, self.form, self.takes_rng, 'ParamCall')
        if self.form.open:
            raise TypeError('ParamCall form cannot be open')

    def copy(self, **changes) -> 'ParamCall':
        return replace(self, **changes)


@dataclass(eq=False)
class InitCall:
    """Canonical state construction, with a distinct priming form.

    Non-priming implementations have signature
    ``impl(definition, param, formed_input, rng) -> state``. Priming
    implementations have signature
    ``impl(definition, param, formed_input, input, rng) -> state``. Runtime
    input is never nullable inside the canonical contract.
    """

    impl: Callable
    form: CallForm
    takes_rng: bool = False
    requires_input: bool = False
    reads_def: bool = False

    def __post_init__(self):
        _check_call(self.impl, self.form, self.takes_rng, 'InitCall')
        if self.form.open:
            raise TypeError('InitCall form cannot be open')
        if type(self.requires_input) is not bool:
            raise TypeError('InitCall.requires_input must be a bool')

    def copy(self, **changes) -> 'InitCall':
        return replace(self, **changes)


@dataclass(eq=False)
class ApplyCall:
    """``impl(definition, param, state, formed_input, rng) -> (state, output)``."""

    impl: Callable
    form: CallForm
    input_spec: PyTree = None
    takes_rng: bool = False
    reads_def: bool = False

    def __post_init__(self):
        _check_call(self.impl, self.form, self.takes_rng, 'ApplyCall')
        from nodejax.binding import REQUIRED
        defaulted = [
            name for name, field in self.form.fields.__items__
            if field.is_nested or field.content is not REQUIRED
        ]
        if defaulted:
            raise TypeError(
                'ApplyCall inputs are always required; defaults declared '
                f'for {defaulted}')

    def copy(self, **changes) -> 'ApplyCall':
        return replace(self, **changes)


@dataclass(eq=False)
class ContractCalls:
    """All canonical executable roles stored by a definition."""

    apply: ApplyCall
    param: ParamCall | None = None
    init: InitCall | None = None

    def __post_init__(self):
        if type(self.apply) is not ApplyCall:
            raise TypeError('ContractCalls.apply must be an ApplyCall')
        if self.param is not None and type(self.param) is not ParamCall:
            raise TypeError('ContractCalls.param must be a ParamCall or None')
        if self.init is not None and type(self.init) is not InitCall:
            raise TypeError('ContractCalls.init must be an InitCall or None')

    @property
    def parametric(self) -> bool:
        return self.param is not None

    @property
    def cyclic(self) -> bool:
        return self.init is not None

    def copy(self, **changes) -> 'ContractCalls':
        return replace(self, **changes)

    def with_param(self, **changes) -> 'ContractCalls':
        if self.param is None:
            return self
        return replace(self, param=replace(self.param, **changes))

    def with_init(self, **changes) -> 'ContractCalls':
        if self.init is None:
            return self
        return replace(self, init=replace(self.init, **changes))

    def with_apply(self, **changes) -> 'ContractCalls':
        return replace(self, apply=replace(self.apply, **changes))


def _check_call(impl: Callable, form: CallForm,
                takes_rng: bool, owner: str) -> None:
    if not callable(impl):
        raise TypeError(f'{owner}.impl must be callable')
    if type(form) is not CallForm:
        raise TypeError(f'{owner}.form must be a CallForm')
    if type(takes_rng) is not bool:
        raise TypeError(f'{owner}.takes_rng must be a bool')


def _empty(value: Any) -> bool:
    return type(value) is tuple and not value


_MISSING_SLOT = object()


def _form_bundle(form: CallForm, supplied: Struct, where: str) -> Struct:
    """Materialize one closed call record from its declaration.

    Public binding reports the friendly argument errors.  This second check is
    the canonical boundary: defaults become explicit fields, including inside
    definition-shaped construction records, so implementations never inspect
    a sparse call bundle.
    """
    from nodejax.binding import REQUIRED

    if not issubclass(type(supplied), Struct):
        raise TypeError(f'{where} expects a formed Struct')
    unknown = set(supplied.__keys__) - set(form.fields.__keys__)
    if unknown and not form.open:
        raise TypeError(f'{where}: unknown fields {sorted(unknown)}')

    values = {}
    for name, field in form.fields.__items__:
        present = name in supplied
        if field.is_nested:
            nested = supplied[name] if present else Struct()
            values[name] = _form_bundle(
                field.content, nested, f'{where}.{name}')
        elif field.content is REQUIRED:
            if not present:
                raise TypeError(f'{where}: missing required field {name!r}')
            values[name] = supplied[name]
        else:
            values[name] = supplied[name] if present else field.content
    if form.open:
        values.update({
            name: value for name, value in supplied.__items__
            if name not in values
        })
    return Struct(**values)


def _form_call(call, supplied: Struct, where: str) -> Struct:
    return _form_bundle(call.form, supplied, where)


def _definition_slots(definition, value: Any, role: str, *,
                      project: bool, path: str = '') -> Any:
    """Convert between sparse public values and dense T4 value trees."""
    active = (definition.parametric if role == 'param' else definition.cyclic)
    if not active:
        if value is not _MISSING_SLOT and not _empty(value):
            raise TypeError(
                f'{path or definition.name}: no {role} value is declared')
        return ()

    if value is _MISSING_SLOT:
        raise TypeError(f'{path or definition.name}: missing {role} value')

    transparent = definition.layout.transparent_member
    if transparent is not None:
        return _definition_slots(
            getattr(definition.members, transparent), value, role,
            project=project, path=path)

    if not definition.members:
        return value

    if _empty(value):
        supplied = Struct()
    elif issubclass(type(value), Struct):
        supplied = value
    else:
        raise TypeError(
            f'{path or definition.name}: composite {role} must be a Struct')

    unknown = set(supplied.__keys__) - set(definition.members.__keys__)
    if unknown:
        raise TypeError(
            f'{path or definition.name}: unknown {role} members '
            f'{sorted(unknown)}')

    values = {}
    param_members = definition.layout.param_members
    for name, member in definition.members.__items__:
        child_path = f'{path}.{name}' if path else name
        slot = (supplied[name] if name in supplied else _MISSING_SLOT)
        child_active = (member.parametric if role == 'param' else
                        member.cyclic)
        if role == 'param' and param_members is not None:
            child_active = child_active and name in param_members
        if child_active:
            converted = _definition_slots(
                member, slot, role, project=project, path=child_path)
        else:
            if slot is not _MISSING_SLOT and not _empty(slot):
                raise TypeError(f'{child_path}: no {role} value is declared')
            converted = ()
        if not project or child_active:
            values[name] = converted
    return Struct(**values)


def _state_tree(definition, fn: Callable) -> Any:
    """Map cyclic leaves into the definition's dense state layout."""
    if not definition.cyclic:
        return ()
    transparent = definition.layout.transparent_member
    if transparent is not None:
        return _state_tree(getattr(definition.members, transparent), fn)
    if definition.members:
        return Struct(**{
            name: _state_tree(member, fn)
            for name, member in definition.members.__items__
        })
    return fn(definition.contract)


def _map_state(definition, state, fn: Callable) -> Any:
    """Map dense state leaves together with their Contract views."""
    if not definition.cyclic:
        return ()
    transparent = definition.layout.transparent_member
    if transparent is not None:
        return _map_state(
            getattr(definition.members, transparent), state, fn)
    if definition.members:
        return Struct(**{
            name: _map_state(member, getattr(state, name), fn)
            for name, member in definition.members.__items__
        })
    return fn(definition.contract, state)


def _boundary_names(definition) -> frozenset[str]:
    names = set(definition.boundaries)
    for _, member in definition.members.__items__:
        names.update(_boundary_names(member))
    return frozenset(names)


def _merge_boundary(definition, carried, initialized,
                    tag: str) -> PyTree:
    """Apply one declared boundary action over a dense state tree."""
    if not definition.cyclic:
        return ()
    transparent = definition.layout.transparent_member
    if transparent is not None:
        child = getattr(definition.members, transparent)
        decided = _merge_boundary(child, carried, initialized, tag)
    elif definition.members:
        decided = Struct(**{
            name: _merge_boundary(
                member, getattr(carried, name), getattr(initialized, name),
                tag)
            for name, member in definition.members.__items__
        })
    else:
        decided = carried
    action = definition.boundaries.get(tag)
    return (decided if action is None else
            action(carried, initialized, decided))


def _role_signature(fn: Callable, role: str):
    signature = inspect.signature(fn)
    if any(parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
    ) for parameter in signature.parameters.values()):
        raise TypeError(
            f'{role} must declare a closed set of named arguments')
    return signature


def _role_arguments(signature, *, definition, formed_input,
                    channels: dict, aggregate: str) -> frozendict:
    available = {
        'contract': definition.contract,
        aggregate: formed_input,
        **channels,
    }
    arguments = {}
    for name, parameter in signature.parameters.items():
        if name in available:
            arguments[name] = available[name]
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"{definition.name}: no contract channel named '{name}'")
    return frozendict(arguments)


def _lower_param(fn: Callable, source: ParamCall) -> ParamCall:
    signature = _role_signature(fn, 'transform param')

    def impl(definition, formed_input, rng):
        return fn(**_role_arguments(
            signature, definition=definition, formed_input=formed_input,
            channels={'rng': rng}, aggregate='param_input'))

    return source.copy(
        impl=impl, reads_def='contract' in signature.parameters)


def _lower_init(fn: Callable, source: InitCall, *, primes: bool) -> InitCall:
    role = 'transform prime' if primes else 'transform init'
    signature = _role_signature(fn, role)

    if primes:
        def impl(definition, param, formed_input, input, rng):
            return fn(**_role_arguments(
                signature, definition=definition,
                formed_input=formed_input,
                channels={'param': param, 'input': input, 'rng': rng},
                aggregate='state_input'))
    else:
        def impl(definition, param, formed_input, rng):
            return fn(**_role_arguments(
                signature, definition=definition,
                formed_input=formed_input,
                channels={'param': param, 'rng': rng},
                aggregate='state_input'))

    return source.copy(
        impl=impl,
        requires_input=primes,
        reads_def='contract' in signature.parameters,
    )


def _lower_apply(fn: Callable, source: ApplyCall) -> ApplyCall:
    signature = _role_signature(fn, 'transform apply')
    has_state = 'state' in signature.parameters

    def impl(definition, param, state, formed_input, rng):
        output = fn(**_role_arguments(
            signature, definition=definition, formed_input=formed_input,
            channels={'param': param, 'state': state, 'rng': rng},
            aggregate='input'))
        return output if has_state else (state, output)

    return source.copy(
        impl=impl, reads_def='contract' in signature.parameters)


class Contract:
    """T3 contract operations projected from one complete definition."""

    def __init__(self, definition):
        from nodejax.definition import Def

        if type(definition) is not Def:
            raise TypeError('Contract expects a Def')
        self._def = definition

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def parametric(self) -> bool:
        return self._def.parametric

    @property
    def cyclic(self) -> bool:
        return self._def.cyclic

    @property
    def members(self) -> Struct:
        """The member contracts, preserving the definition's Struct shape."""
        return Struct(**{
            name: member.contract
            for name, member in self._def.members.__items__
        })

    @property
    def tags(self) -> frozenset[str]:
        return self._def.tags

    def _roles(self, *, param=None, init=None, prime=None, apply=None,
               param_takes_rng=None, init_takes_rng=None,
               apply_takes_rng=None, input_spec=_KEEP,
               apply_fields=None, open: bool = False):
        """Lower ordinary Python role functions over this contract.

        Functions name only the channels they use. ``contract`` receives the
        current contract; ``param_input`` and ``state_input`` receive formed
        constructor records; apply receives the complete formed call bundle
        as ``input``. Prime receives the real runtime value under the same
        name. This contract decides which roles exist, whether init or prime
        applies, their call forms, and their public RNG requirements.
        ``init=False`` removes state from the result. ``apply_fields`` declares
        a replacement required-field form; ``open`` also permits undeclared
        side fields.
        """
        calls = self._def.calls
        if calls.param is not None and param is not None:
            calls = calls.copy(param=_lower_param(param, calls.param))
        if init is False:
            if prime is not None:
                raise TypeError('init=False cannot be combined with prime=')
            calls = calls.copy(init=None)
        elif calls.init is not None:
            initializer = prime if calls.init.requires_input else init
            if initializer is not None:
                calls = calls.copy(init=_lower_init(
                    initializer, calls.init,
                    primes=calls.init.requires_input))
        if apply is not None:
            calls = calls.copy(apply=_lower_apply(apply, calls.apply))

        overrides = (
            ('param', param_takes_rng),
            ('init', init_takes_rng),
            ('apply', apply_takes_rng),
        )
        for role, value in overrides:
            if value is None:
                continue
            if type(value) is not bool:
                raise TypeError(f'{role}_takes_rng must be a bool or None')
            call = getattr(calls, role)
            if call is None:
                raise TypeError(
                    f'{role}_takes_rng cannot describe an absent role')
            calls = calls.copy(**{role: call.copy(takes_rng=value)})

        if apply_fields is not None:
            from nodejax.binding import REQUIRED

            calls = calls.with_apply(
                form=CallForm.from_values(Struct(**{
                    field: REQUIRED for field in apply_fields}), open=open),
                input_spec=(None if input_spec is _KEEP else input_spec),
            )
        elif input_spec is not _KEEP:
            calls = calls.with_apply(input_spec=input_spec)
        return calls

    @property
    def param_takes_rng(self) -> bool:
        call = self._def.calls.param
        return False if call is None else call.takes_rng

    @property
    def init_takes_rng(self) -> bool:
        call = self._def.calls.init
        return False if call is None else call.takes_rng

    @property
    def apply_takes_rng(self) -> bool:
        return self._def.calls.apply.takes_rng

    @property
    def _apply_form(self) -> CallForm:
        return self._def.calls.apply.form

    @property
    def apply_fields(self) -> tuple[str, ...]:
        return tuple(self._apply_form.declaration.__keys__)

    @property
    def _accepts_input(self) -> bool:
        return self._apply_form.open or bool(self.apply_fields)

    @property
    def input_spec(self):
        from nodejax.binding import _spec_resolved

        spec = self._def.calls.apply.input_spec
        return spec if _spec_resolved(spec) else None

    def input_spec_for(self, field: str):
        """Resolved input evidence for one call field, or ``None``."""
        spec = self.input_spec
        return None if spec is None else spec[field]

    @property
    def param_input_spec(self) -> Struct | None:
        call = self._def.calls.param
        return None if call is None else call.form.declaration

    @property
    def state_input_spec(self) -> Struct | None:
        call = self._def.calls.init
        return None if call is None else call.form.declaration

    @property
    def init_requires_input(self) -> bool:
        call = self._def.calls.init
        return False if call is None else call.requires_input

    def _resolve_def(self, input_spec, *, replace_evidence: bool = False,
                     bundled: bool = False):
        """Resolve one wire, or an explicitly formed bundle when requested."""
        from nodejax.binding import (
            _bind_axis, _contains_axis, _counts_unknown, _spec_resolved,
            _validate_spec, validate_input_spec,
        )
        from nodejax.spec import spec_of

        evidence = spec_of(input_spec)
        got = (self._apply_form.feed_bundle(evidence) if bundled else
               self._apply_form.feed(evidence))
        if issubclass(type(got), Struct) and 'rng' in got:
            raise TypeError(
                f'{self._def.name}.with_input: rng is a separate call channel')
        declared = self._def.calls.apply.input_spec
        if (not replace_evidence and self.input_spec is not None):
            validate_input_spec(self._def, got)
            if not _counts_unknown(declared):
                return self._def
        elif _contains_axis(declared) and _spec_resolved(declared):
            _validate_spec(self._def.name, declared, got)
        resolved = _bind_axis(declared, got)
        return self._def.copy(
            calls=self._def.calls.with_apply(input_spec=resolved))

    def _resolve(self, input_spec, *, replace_evidence: bool):
        from nodejax.node import _view

        return _view(self._resolve_def(
            input_spec, replace_evidence=replace_evidence))

    def with_input(self, input_spec):
        """Replace this definition's input-shape evidence."""
        return self._resolve(input_spec, replace_evidence=True)

    def resolve(self, input_spec):
        """Bind missing input evidence or validate existing evidence."""
        return self._resolve(input_spec, replace_evidence=False)

    def for_input(self, input_spec) -> 'Contract':
        """Use this contract with optional input-shape evidence."""
        return (self if input_spec is None else
                self._resolve_def(input_spec).contract)

    def feed(self, value: PyTree) -> PyTree:
        """Format one computed value for this contract's apply call."""
        return self._apply_form.feed(value)

    def feed_bundle(self, value: PyTree) -> Struct:
        """Accept one explicitly formed call bundle without adaptation."""
        return self._apply_form.feed_bundle(value)

    def intake(self, value: Struct) -> PyTree:
        """Recover this contract's wire from one formed apply bundle."""
        return self._apply_form.intake(value)

    def _dense_param(self, value):
        """Form this definition's fixed T4 parameter tree."""
        return _definition_slots(self._def, value, 'param', project=False)

    def _sparse_param(self, value):
        """Project a canonical parameter tree for a public Node view."""
        return _definition_slots(self._def, value, 'param', project=True)

    def _dense_state(self, value):
        """Form this definition's fixed T4 state tree."""
        return _definition_slots(self._def, value, 'state', project=False)

    def _sparse_state(self, value):
        """Project a canonical state tree for a public Node view."""
        return _definition_slots(self._def, value, 'state', project=True)

    def member_param(self, name: str, formed_input, rng: MaybeKeyStream, *,
                     input_spec=None):
        """Construct one member parameter slot, honoring bound captures."""
        if name not in self._def.members:
            raise TypeError(f'{self.name}: no member {name!r}')
        member = getattr(self._def.members, name)
        if not member.parametric:
            return ()
        if name in self._def.captures.param:
            return formed_input
        child = member.contract.for_input(input_spec)
        return child.param(
            formed_input, rng.child(child.param_takes_rng))

    def member_init(self, name: str, param, formed_input,
                    rng: MaybeKeyStream, *, input=_KEEP,
                    input_spec=None):
        """Construct one member state slot, honoring bound captures."""
        if name not in self._def.members:
            raise TypeError(f'{self.name}: no member {name!r}')
        member = getattr(self._def.members, name)
        if not member.cyclic:
            return ()
        if name in self._def.captures.state:
            from nodejax.binding import _has_rng_deep
            if _has_rng_deep(self._def.captures.state[name]):
                from nodejax.compose import _rekeyed
                return _rekeyed(
                    formed_input, rng.next(),
                    f"member '{name}'")
            return formed_input
        child = member.contract.for_input(input_spec)
        child_rng = rng.child(child.init_takes_rng)
        member_param = param if member.parametric else ()
        if input is _KEEP:
            return child.init(member_param, formed_input, child_rng)
        return child.prime(
            member_param, formed_input, input, child_rng)

    def state_tree(self, fn: Callable) -> Any:
        """Map a value over cyclic leaves in the canonical state layout."""
        return _state_tree(self._def, fn)

    def map_state(self, state, fn: Callable) -> Any:
        """Map ``fn(contract, value)`` over canonical state leaves."""
        return _map_state(self._def, self._dense_state(state), fn)

    @property
    def boundary_names(self) -> frozenset[str]:
        """Boundary names declared by this definition or its members."""
        return _boundary_names(self._def)

    def merge_boundary(self, carried, initialized, tag: str):
        """Apply matching boundary actions to canonical state trees."""
        return _merge_boundary(
            self._def,
            self._dense_state(carried),
            self._dense_state(initialized),
            tag,
        )

    @staticmethod
    def _check_rng(rng: MaybeKeyStream, takes_rng: bool, role: str) -> None:
        if type(rng) is not MaybeKeyStream:
            raise TypeError(f'contract.{role} expects a MaybeKeyStream')
        if bool(rng) != takes_rng:
            expected = 'keyed' if takes_rng else 'empty'
            raise TypeError(
                f'contract.{role} expects a {expected} MaybeKeyStream')

    def param(self, formed_input: Struct, rng: MaybeKeyStream):
        if not issubclass(type(formed_input), Struct):
            raise TypeError('contract.param expects a formed Struct')
        call = self._def.calls.param
        if call is None:
            self._check_rng(rng, False, 'param')
            return ()
        self._check_rng(rng, call.takes_rng, 'param')
        formed_input = _form_call(
            call, formed_input, f'{self._def.name}.param')
        return self._dense_param(call.impl(self._def, formed_input, rng))

    def init(self, param, formed_input: Struct, rng: MaybeKeyStream):
        if not issubclass(type(formed_input), Struct):
            raise TypeError('contract.init expects a formed Struct')
        call = self._def.calls.init
        if call is None:
            self._check_rng(rng, False, 'init')
            return ()
        if call.requires_input:
            raise TypeError(
                f'{self._def.name} requires a real input value; '
                'use contract.prime')
        self._check_rng(rng, call.takes_rng, 'init')
        param = self._dense_param(param)
        formed_input = _form_call(
            call, formed_input, f'{self._def.name}.init')
        return self._dense_state(
            call.impl(self._def, param, formed_input, rng))

    def prime(self, param, formed_input: Struct,
              input, rng: MaybeKeyStream):
        if not issubclass(type(formed_input), Struct):
            raise TypeError('contract.prime expects a formed Struct')
        definition = self._resolve_def(input)
        contract = definition.contract
        call = contract._def.calls.init
        if call is None:
            self._check_rng(rng, False, 'prime')
            return ()
        self._check_rng(rng, call.takes_rng, 'prime')
        param = contract._dense_param(param)
        formed_input = _form_call(
            call, formed_input, f'{definition.name}.init')
        if call.requires_input:
            state = call.impl(
                definition, param, formed_input, input, rng)
        else:
            state = call.impl(definition, param, formed_input, rng)
        return contract._dense_state(state)

    def apply(self, param, state, formed_input: Struct,
              rng: MaybeKeyStream):
        if not issubclass(type(formed_input), Struct):
            raise TypeError('contract.apply expects a formed Struct')
        call = self._def.calls.apply
        self._check_rng(rng, call.takes_rng, 'apply')
        formed_input = _form_call(
            call, formed_input, f'{self._def.name}.apply')
        next_state, output = call.impl(
            self._def,
            self._dense_param(param),
            self._dense_state(state),
            formed_input,
            rng,
        )
        return self._dense_state(next_state), output
