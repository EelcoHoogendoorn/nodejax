"""Serial and authored composition over recursive definitions."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import re
from functools import wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp

from nodejax.core.ambient import node
from nodejax.core.binding import (
    Aux, _bundle_spec_from_sig, _has_rng_deep, _spec_resolved,
    split_aux,
)
from nodejax.core.composite import (
    _promote_members, member_param_input, member_state_input,
)
from nodejax.core.contract import (
    ApplyCall, CallForm, ContractCalls, InitCall, ParamCall,
)
from nodejax.core.definition import Captures, Construction, Def, Layout
from nodejax.frozendict import frozendict
from nodejax.core.generic import Generic, is_generic
from nodejax.core.lifting import (
    _check_methods, _compile_init, _compile_param,
)
from nodejax.core.node import BaseNode, Node, _is_node, _view
from nodejax.core.rng import MaybeKeyStream
from nodejax.core.spec import materialize, spec_of
from nodejax.struct import Struct
from nodejax.tree import tree_first
from nodejax.core.wiring import (
    _BuildingWired, _InitWired, _authored,
)


_EMPTY_MAPPING = frozendict()


def apply_call(impl: Callable, form: CallForm, *, input_spec=None,
               takes_rng: bool = False) -> ApplyCall:
    return ApplyCall(
        impl=impl,
        form=form,
        input_spec=input_spec if _spec_resolved(input_spec) else None,
        takes_rng=takes_rng,
    )


def param_call(impl: Callable, form: CallForm, *,
               takes_rng: bool = False,
               reads_def: bool = True) -> ParamCall:
    return ParamCall(impl, form, takes_rng, reads_def)


def init_call(impl: Callable, form: CallForm, *,
              requires_input: bool = False,
              takes_rng: bool = False,
              reads_def: bool = True) -> InitCall:
    return InitCall(impl, form, takes_rng, requires_input, reads_def)


def _probe_rng(takes_rng: bool) -> MaybeKeyStream:
    """Private entropy for an abstract execution whose values are discarded."""
    return (MaybeKeyStream(jax.random.PRNGKey(0)) if takes_rng
            else MaybeKeyStream())


def _param_takes_rng(members: Struct, captures: Captures) -> bool:
    return any(
        name not in captures.param and member.contract.param_takes_rng
        for name, member in members.__items__)


def _init_takes_rng(members: Struct, captures: Captures) -> bool:
    return any(
        (_has_rng_deep(captures.state[name]) if name in captures.state
         else member.contract.init_takes_rng)
        for name, member in members.__items__)


def _wiring_takes_rng(members: Struct, authored,
                      rng_from: bool | None = None) -> bool:
    return (authored.author_rng and rng_from is not False) or any(
        member.contract.apply_takes_rng for member in members)


def block_name(separator: str, names) -> str:
    return '(' + separator.join(names) + ')'


def _over_members(separator: str):
    """Normalize an N-ary construction into member Defs plus captures."""
    def decorate(build):
        @wraps(build)
        def entry(**named):
            if not named:
                raise TypeError(f'{build.__name__} needs at least one member')
            if any(is_generic(value) for value in named.values()):
                return Generic(
                    block_name(separator, named),
                    lambda **members: entry(**members),
                    Struct(**named),
                )
            definitions, captures = _promote_members(named)
            product = build(definitions, captures)
            if not _is_node(product):
                raise TypeError('member combinator did not build a Node')

            def bind(replacements):
                return build(replacements, Captures())._def

            definition = product._def.copy(
                tree=bind,
                construction=Construction(
                    entry,
                    Struct(**{name: Node(definition)
                              for name, definition in definitions.__items__}),
                ),
            )
            product = _view(definition)

            if all(value.bound for value in named.values()):
                param = Struct(**{
                    name: named[name].param
                    for name, member in definitions.__items__
                    if member.parametric
                })
                cyclic = [name for name, definition in definitions.__items__
                          if definition.cyclic]
                if cyclic and all(name in captures.state for name in cyclic):
                    state = Struct(**{
                        name: value.state for name, value in named.items()
                        if getattr(definitions, name).cyclic
                    })
                    return Node(definition).bind(
                        param if param else (), state=state if state else ())
                return Node(definition).bind(param if param else ())
            return product

        entry.separator = separator
        return entry
    return decorate


def _rekeyed(state, key, where):
    draws = 0

    def walk(value, current):
        nonlocal draws
        if not issubclass(type(value), Struct):
            return value, current
        out = {}
        for name, child in value.__items__:
            if name != 'rng':
                out[name], current = walk(child, current)
                continue
            current, fresh = jax.random.split(current)
            draws += 1
            try:
                template = jax.random.key_data(child)
                shape = template.shape[:-1]
                count = 1
                for extent in shape:
                    count *= extent
                keys = jax.random.split(fresh, count) if shape else fresh
                data = jax.random.key_data(keys).reshape(shape + template.shape[-1:])
                out[name] = (
                    jax.random.wrap_key_data(data, impl=jax.random.key_impl(child))
                    if jax.dtypes.issubdtype(
                        getattr(child, 'dtype', None), jax.dtypes.prng_key)
                    else data)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{where}: state rng is not a JAX key") from error
        return Struct(**out), current

    result, _ = walk(state, key)
    if not draws:
        raise TypeError(f'{where}: replacement state contains no rng field')
    return result


def _probe_apply(apply_fn, param, state, input, rng):
    bound_axes = set()
    call = lambda p, s, i: apply_fn(p, s, i, rng)
    while True:
        try:
            return call(param, state, input)
        except NameError as error:
            match = re.search(r'unbound axis name:\s*(\w+)', str(error))
            if not match or match.group(1) in bound_axes:
                raise
            axis = match.group(1)
            bound_axes.add(axis)
            previous = call

            def call(p, s, i, previous=previous, axis=axis):
                lead = lambda tree: jax.tree.map(
                    lambda value: jnp.asarray(value)[None], tree)
                return tree_first(jax.vmap(previous, axis_name=axis)(
                    lead(p), lead(s), lead(i)))


def _step_priming_carry(member, param, state, carry, name, rng):
    formed_input = (member.contract.feed(carry)
                    if member.contract._accepts_input else Struct())
    try:
        _, output = _probe_apply(
            member.contract.apply, param, state,
            formed_input, rng)
    except Exception as error:
        raise TypeError(f"walk failed at member '{name}': {error}") from error
    value, aux = split_aux(output)
    return value


def _step_spec(member, param, state, spec, name):
    from nodejax.core.spec import _axis_probe_spec
    formed_input = (member.contract.feed(_axis_probe_spec(spec))
                    if member.contract._accepts_input else Struct())
    try:
        output = jax.eval_shape(
            lambda member_param, member_state, member_input: _probe_apply(
                member.contract.apply,
                member_param,
                member_state,
                member_input,
                _probe_rng(member.contract.apply_takes_rng))[1],
            param, state, formed_input,
        )
    except Exception as error:
        raise TypeError(f"spec walk failed at member '{name}': {error}") from error
    value, aux = split_aux(output)
    return value


def _member_param(name, member, sub_bundle, rng, input_spec=None, *,
                  bundled: bool = False):
    if not member.parametric:
        return ()
    bundle = Struct() if sub_bundle is None else sub_bundle
    current = (member if input_spec is None else
               member.contract._resolve_def(input_spec, bundled=bundled))
    try:
        return current.contract.param(bundle, rng)
    except TypeError as error:
        raise TypeError(f"member '{name}': {error}") from error


def _member_states_init(members, captures: Captures, *, block, project):
    if not any(member.cyclic for member in members):
        return None
    requires_input = any(
        member.contract.init_requires_input and name not in captures.state
        for name, member in members.__items__)
    takes_rng = _init_takes_rng(members, captures)

    def initialize(definition, param, formed_input, rng, value=None):
        current = definition.members
        captures = definition.captures
        states = {}
        for name, member in current.__items__:
            if not member.cyclic:
                continue
            if name in captures.state:
                supplied = (formed_input[name] if name in formed_input
                            else captures.state[name])
                if _has_rng_deep(captures.state[name]):
                    supplied = _rekeyed(
                        supplied, rng.next(),
                        f"{block} member '{name}'")
                states[name] = supplied
                continue
            evidence = project(value, name) if value is not None else None
            if evidence is None and definition.contract.input_spec is not None:
                evidence = project(definition.contract.input_spec, name)
            resolved = (member if evidence is None else
                        member.contract._resolve_def(evidence))
            child = rng.child(resolved.contract.init_takes_rng)
            state_input = getattr(formed_input, name)
            params = (getattr(param, name) if member.parametric else ())
            states[name] = (resolved.contract.prime(
                params, state_input, evidence, child)
                if value is not None else
                resolved.contract.init(params, state_input, child))
        return Struct(**states)

    if requires_input:
        def impl(definition, param, formed_input, input, rng):
            return initialize(definition, param, formed_input, rng, input)
    else:
        def impl(definition, param, formed_input, rng):
            return initialize(definition, param, formed_input, rng)
    return init_call(
        impl, member_state_input(members, captures),
        requires_input=requires_input, takes_rng=takes_rng)


@_over_members(' >> ')
def serial(members: Struct, captures: Captures):
    names = members.__keys__
    last_primer = max((index for index, (name, member)
                       in enumerate(members.__items__)
                       if member.contract.init_requires_input
                       and name not in captures.state), default=-1)
    requires_input = last_primer >= 0
    param_takes_rng = _param_takes_rng(members, captures)
    init_takes_rng = _init_takes_rng(members, captures) or any(
        getattr(members, names[index]).contract.apply_takes_rng
        for index in range(last_primer))

    def param_impl(definition, formed_input, rng):
        current = definition.members
        captures = definition.captures
        input_spec = definition.contract.input_spec
        carry = (None if input_spec is None else
                 definition.contract.intake(input_spec))
        parts = {}
        for name, member in current.__items__:
            supplied = formed_input[name]
            accepts_input = member.contract._accepts_input
            if member.parametric:
                if name in captures.param:
                    parts[name] = supplied
                else:
                    parts[name] = _member_param(
                        name, member, supplied,
                        rng.child(member.contract.param_takes_rng),
                        input_spec=carry if accepts_input else None)
            if carry is not None:
                resolved = (member.contract._resolve_def(carry)
                            if accepts_input else member)
                child = _probe_rng(resolved.contract.init_takes_rng)
                state = (resolved.contract.prime(
                    parts.get(name, ()), Struct(), materialize(carry), child)
                    if resolved.contract.init_requires_input else
                    resolved.contract.init(
                        parts.get(name, ()), Struct(), child))
                carry = _step_spec(
                    resolved, parts.get(name, ()), state, carry, name)
        return Struct(**parts) if parts else ()

    def init_walk(definition, param, formed_input, rng, value=None):
        current = definition.members
        captures = definition.captures
        current_items = tuple(current.__items__)
        last_cyclic = max((index for index, (name, member)
                           in enumerate(current_items)
                           if member.cyclic
                           and name not in captures.state),
                          default=-1)
        real = value is not None
        input_spec = definition.contract.input_spec
        if real:
            carry = value
        elif input_spec is None:
            carry = None
        else:
            carry = definition.contract.intake(input_spec)
        shape = spec_of(carry) if real else carry
        states = {}
        for index, (name, member) in enumerate(current_items):
            evidence = carry if real and index <= last_primer else shape
            member_evidence = (evidence
                               if member.contract._accepts_input else None)
            if member.cyclic:
                if name in captures.state:
                    state = (formed_input[name] if name in formed_input
                             else captures.state[name])
                    if _has_rng_deep(captures.state[name]):
                        state = _rekeyed(
                            state, rng.next(),
                            f"pipe member '{name}'")
                    states[name] = state
                else:
                    resolved = (member if member_evidence is None else
                                member.contract._resolve_def(member_evidence))
                    child = rng.child(resolved.contract.init_takes_rng)
                    bundle = getattr(formed_input, name)
                    params = (getattr(param, name)
                              if member.parametric else ())
                    states[name] = (resolved.contract.prime(
                        params, bundle, member_evidence, child)
                        if (real and index <= last_primer
                            and member_evidence is not None) else
                        resolved.contract.init(params, bundle, child))
            if index >= last_cyclic or evidence is None:
                continue
            if real and index < last_primer:
                carry = _step_priming_carry(
                    member, (getattr(param, name)
                             if member.parametric else ()),
                    states.get(name, ()), carry, name,
                    rng.child(member.contract.apply_takes_rng))
                shape = spec_of(carry)
            else:
                shape = _step_spec(
                    member, (getattr(param, name)
                             if member.parametric else ()),
                    states.get(name, ()), shape, name)
        return Struct(**states) if states else ()

    if requires_input:
        def init_impl(definition, param, formed_input, input, rng):
            return init_walk(definition, param, formed_input, rng, input)
    else:
        def init_impl(definition, param, formed_input, rng):
            return init_walk(definition, param, formed_input, rng)

    apply_takes_rng = any(
        member.contract.apply_takes_rng for member in members)

    def apply_impl(definition, param, state, formed_input, rng):
        current = definition.members
        carry = definition.contract.intake(formed_input)
        states, aux = {}, {}
        for name, member in current.__items__:
            child = rng.child(member.contract.apply_takes_rng)
            member_input = (member.contract.feed(carry)
                            if member.contract._accepts_input else Struct())
            next_state, output = member.contract.apply(
                (getattr(param, name) if member.parametric else ()),
                (getattr(state, name) if member.cyclic else ()),
                member_input, child)
            if member.cyclic:
                states[name] = next_state
            carry, member_aux = split_aux(output)
            if member_aux is not None:
                aux[name] = member_aux
        result_state = Struct(**states) if states else ()
        return (result_state, (carry, Aux(**aux))) if aux else (
            result_state, carry)

    head = getattr(members, names[0])
    calls = ContractCalls(
        param=(param_call(
            param_impl,
            member_param_input(members, captures),
            takes_rng=param_takes_rng) if any(member.parametric
                                              for member in members) else None),
        init=(init_call(
            init_impl, member_state_input(members, captures),
            requires_input=requires_input, takes_rng=init_takes_rng)
              if any(member.cyclic for member in members) else None),
        apply=apply_call(
            apply_impl, head.calls.apply.form,
            input_spec=head.contract.input_spec,
            takes_rng=apply_takes_rng),
    )
    return _view(Def(
        name=block_name(' >> ', names), calls=calls,
        members=members,
        captures=captures, layout=Layout(kind='serial')))


def _member_init(members, authored, param, evidence, rng, inputs, *,
                 prime, captures):
    if evidence is not None and authored.wired:
        wired = _InitWired(
            members, param, rng, inputs, dict(captures.state),
            real=prime, author_rng=prime and authored.author_rng)
        if prime:
            authored.run(wired, materialize(evidence))
        else:
            jax.eval_shape(lambda value: authored.run(wired, value), evidence)
        return wired._collect()

    states = {}
    for name, member in members.__items__:
        if name in captures.state and member.cyclic:
            supplied = (getattr(inputs, name) if name in inputs
                        else captures.state[name])
            if _has_rng_deep(captures.state[name]):
                supplied = _rekeyed(
                    supplied, rng.next(),
                    f"member '{name}'")
            states[name] = supplied
            continue
        child = rng.child(member.contract.init_takes_rng)
        bundle = getattr(inputs, name)
        state = (member.contract.prime(
            (getattr(param, name) if member.parametric else ()),
            bundle, evidence, child)
            if prime and member.contract.init_requires_input else
            member.contract.init(
                (getattr(param, name) if member.parametric else ()),
                bundle, child))
        if member.cyclic:
            states[name] = state
    return Struct(**states) if states else ()


def _checked_init(call: InitCall, members, name):
    def validate(state):
        from nodejax.core.contract import _empty
        expected = {
            field for field, member in members.__items__ if member.cyclic}
        entries = (dict(state.__items__)
                   if issubclass(type(state), Struct) else {})
        # An explicit empty slot for a stateless member is permitted and
        # stripped here, so authored inits need not fork on member
        # lifecycles while the stored state stays canonically sparse.
        vacuous = {
            field for field, value in entries.items()
            if field not in expected
            and field in set(members.__keys__)
            and _empty(value)}
        got = set(entries) - vacuous
        if got != expected:
            raise TypeError(
                f'{name or "composite"}: init state keys {sorted(got)}; '
                f'expected {sorted(expected)}')
        if vacuous:
            return Struct(**{
                field: value for field, value in entries.items()
                if field not in vacuous})
        return state
    if call.requires_input:
        def impl(definition, param, formed_input, input, rng):
            return validate(call.impl(
                definition, param, formed_input, input, rng))
    else:
        def impl(definition, param, formed_input, rng):
            return validate(call.impl(definition, param, formed_input, rng))
    return call.copy(impl=impl)


@node
def composite(apply: Callable, *, members: dict[str, BaseNode], param=None,
              init=None, apply_input_spec=None, name=None,
              methods: Mapping = _EMPTY_MAPPING,
              rng_from: bool | None = None, scope: Callable | None = None):
    """``rng_from`` is the authored rng policy (see ``_Wired``) and ``scope``
    the author's view of each walk object; both serve the one-member
    wrapper, which is this composite without a nesting level."""
    definitions, captures = _promote_members(members)
    reserved = {'param', 'state'} & set(definitions.__keys__)
    if reserved:
        raise TypeError(f'member names shadow self fields: {sorted(reserved)}')
    authored = _authored(apply, scope=scope)
    marker = (_bundle_spec_from_sig(apply, drop=('self',))
              if authored.fields else None)
    author_rng = marker is not None and 'rng' in marker
    lifted = authored.lift(definitions, rng_from=rng_from)
    self_form = authored.wired

    def apply_impl(definition, param, state, formed_input, rng):
        return lifted(definition, param, state, formed_input, rng)

    def generated_param(definition, formed_input, rng):
        current = definition.members
        captures = definition.captures
        slots = {field: formed_input[field]
                 for field in current.__keys__ if field in formed_input}
        shape = definition.contract.input_spec
        if (shape is not None and self_form and any(
                member.parametric and member.calls.param.reads_def
                for member in current)):
            wired = _BuildingWired(
                current, dict(captures.param), rng, slots)
            jax.eval_shape(lambda value: authored.run(wired, value), shape)
            values = wired.parameters()
            return values
        values = {}
        for field, member in current.__items__:
            if not member.parametric:
                continue
            supplied = slots.get(field)
            if field in captures.param:
                values[field] = slots[field]
            else:
                values[field] = _member_param(
                    field, member, supplied,
                    rng.child(member.contract.param_takes_rng))
        return Struct(**values)

    def generated_init(definition, param, formed_input, rng,
                       evidence=None, prime=False):
        current = definition.members
        captures = definition.captures
        inputs = formed_input
        if evidence is None and any(
                field not in captures.state
                and member.cyclic and member.calls.init.reads_def
                for field, member in current.__items__):
            evidence = definition.contract.input_spec
        elif evidence is not None and prime and self_form:
            evidence = definition.contract.feed(evidence)
        return _member_init(
            current, authored, param, evidence, rng, inputs,
            prime=prime, captures=captures)

    any_param = any(member.parametric for member in definitions)
    if param is not None and not any_param:
        raise TypeError('a composite has no parameters outside its members')
    custom_param = _compile_param(param) if param is not None else None
    custom_init = _compile_init(init) if init is not None else None
    cyclic = any(member.cyclic for member in definitions) or init is not None
    requires_input = (custom_init.requires_input if custom_init else any(
        member.contract.init_requires_input and field not in captures.state
        for field, member in definitions.__items__))

    if requires_input and custom_init is None:
        def init_impl(definition, param, formed_input, input, rng):
            return generated_init(
                definition, param, formed_input, rng,
                evidence=input, prime=True)
    elif custom_init is None:
        def init_impl(definition, param, formed_input, rng):
            return generated_init(definition, param, formed_input, rng)

    param_role = (custom_param if custom_param else param_call(
        generated_param,
        member_param_input(definitions, captures),
        takes_rng=_param_takes_rng(definitions, captures))) if any_param else None
    init_role = None
    if cyclic:
        init_role = custom_init or init_call(
            init_impl, member_state_input(definitions, captures),
            requires_input=requires_input,
            takes_rng=(
                _init_takes_rng(definitions, captures)
                or (requires_input and self_form
                    and _wiring_takes_rng(definitions, authored, rng_from))))
        if custom_init is not None:
            init_role = _checked_init(custom_init, definitions, name)

    form = (CallForm.from_values(
                marker.without('rng') if author_rng else marker)
            if marker is not None else
            CallForm.from_values(Struct(), open=True))
    input_spec = (None if apply_input_spec is None else
                  form.feed(apply_input_spec))
    checked_methods = _check_methods(methods)
    definition = Def(
        name=name or f'composite({", ".join(definitions.__keys__)})',
        calls=ContractCalls(
            param=param_role,
            init=init_role,
            apply=apply_call(
                apply_impl, form, input_spec=input_spec,
                takes_rng=(_wiring_takes_rng(definitions, authored, rng_from)
                           if self_form else False)),
        ),
        members=definitions,
        captures=captures,
        methods=checked_methods,
        layout=Layout(kind='composite'),
    )

    def bind(replacements):
        rebuilt = composite(
            apply, members={field: Node(child)
                            for field, child in replacements.__items__},
            param=param, init=init, apply_input_spec=apply_input_spec,
            name=name, methods=checked_methods, rng_from=rng_from,
            scope=scope)
        return rebuilt._def

    return _view(definition.copy(tree=bind))


def _defer_wrapper(build):
    @wraps(build)
    def entry(apply, operand, *, member, init=None, name=None, rng_from=None,
              input_spec=None):
        if is_generic(operand):
            return Generic(
                name or f'wrapper({operand.name})',
                lambda **filled: entry(
                    apply, filled[member], member=member, init=init,
                    name=name, rng_from=rng_from, input_spec=input_spec),
                Struct(**{member: operand}),
            )
        return build(apply, operand, member=member, init=init, name=name,
                     rng_from=rng_from, input_spec=input_spec)
    return entry


@_defer_wrapper
@node
def _wrap_build(apply, operand: BaseNode, *, member, init=None, name=None,
                rng_from: bool | None = None, input_spec=None):
    """An authored wrapper is the one-member composite without a nesting
    level: the composite builds, initializes, and runs the member; this
    presents its keyed param, state, and aux as the member's own.
    ``input_spec`` is the wrapper's own declared input wire, as for a
    composite; a wrapper publishes nothing it is not told."""
    from nodejax.core.wrapper import _transparent_def

    if not _authored(apply).wired:
        raise TypeError('wrapper apply must call its named member through self')
    if rng_from is not None and 'rng' not in _bundle_spec_from_sig(
            apply, drop=('self',)):
        raise TypeError('rng_from requires an rng parameter')
    child = operand._def
    # Over a stateless member a declared init is vacuous: the empty state is
    # the only state such a subtree has, so the declaration drops and
    # callers need not fork on the member's lifecycle.
    declared_init = (None if init is None or child.calls.init is None
                     else _transparent_init(init, member))
    inner = composite(
        apply, members={member: operand}, init=declared_init,
        apply_input_spec=input_spec, name=name or f'wrapper({child.name})',
        rng_from=rng_from, scope=lambda wired: _TransparentScope(wired, member),
    )._def

    def keyed(value):
        return Struct(**{member: value})

    calls = inner.calls
    if calls.param is not None:
        inner_param = calls.param

        def param_impl(definition, formed_input, rng):
            return getattr(
                inner_param.impl(definition, keyed(formed_input), rng), member)

        calls = calls.with_param(impl=param_impl, form=child.calls.param.form)
    if calls.init is not None:
        inner_init = calls.init
        # A generated init takes the member's bundle under its name; a
        # declared one takes the wrapper's own fields as they are.
        generated = declared_init is None
        formed = keyed if generated else (lambda formed_input: formed_input)
        if inner_init.requires_input:
            def init_impl(definition, param, formed_input, input, rng):
                return getattr(inner_init.impl(
                    definition, keyed(param), formed(formed_input), input, rng,
                ), member)
        else:
            def init_impl(definition, param, formed_input, rng):
                return getattr(inner_init.impl(
                    definition, keyed(param), formed(formed_input), rng,
                ), member)
        calls = calls.with_init(
            impl=init_impl,
            form=child.calls.init.form if generated else inner_init.form)
    inner_apply = calls.apply

    def apply_impl(definition, param, state, formed_input, rng):
        next_state, output = inner_apply.impl(
            definition, keyed(param), keyed(state), formed_input, rng)
        value, aux = split_aux(output)
        entries = dict(aux.__items__) if type(aux) is Aux else {}
        member_aux = entries.pop(member, None)
        flattened = {
            **(dict(member_aux.__items__) if member_aux is not None else {}),
            **entries,
        }
        return getattr(next_state, member), (
            (value, Aux(**flattened)) if flattened else value)

    def bind(replacements):
        rebuilt = _wrap_build(
            apply, Node(replacements[member]), member=member, init=init,
            name=name, rng_from=rng_from, input_spec=input_spec)
        return rebuilt._def

    definition = _transparent_def(
        member, child, name=inner.name, captures=inner.captures, tags=child.tags,
    ).copy(calls=calls.with_apply(impl=apply_impl), tree=bind)
    return operand._with_definition(definition)


def _transparent_init(init: Callable, member: str) -> Callable:
    """A wrapper's declared init, written against the member's own param
    and returning the member's own state, in the composite's keyed form."""
    def keyed(**arguments):
        if 'param' in arguments:
            arguments['param'] = getattr(arguments['param'], member)
        return Struct(**{member: init(**arguments)})

    keyed.__signature__ = inspect.signature(init)
    return keyed


class _TransparentScope:
    """The wrapper's ``self``: a one-member walk object whose ``param`` and
    ``state`` are the member's own slot rather than a keyed Struct."""

    def __init__(self, wired, member: str):
        self._wired = wired
        self._member = member

    @property
    def param(self):
        return getattr(self._wired.param, self._member)

    @property
    def state(self):
        return getattr(self._wired.state, self._member)

    def __getattr__(self, name):
        return getattr(self._wired, name)


def _ident(name):
    return re.sub(r'\W+', '_', name).strip('_') or 'node'


def _components(value):
    if is_generic(value) or value.state_bound:
        return frozendict({_ident(value.name): value})
    definition = value._def
    if definition.layout.kind == 'serial':
        if value.bound:
            return frozendict({
                name: Node(member).bind(
                    getattr(value.param, name) if member.parametric else ())
                for name, member in definition.members.__items__
            })
        return frozendict({
            name: Node(member) for name, member in definition.members.__items__
        })
    return frozendict({_ident(value.name): value})


def _compose(left, right):
    if not is_generic(left) and not _is_node(left):
        raise TypeError('left side of >> must be a Node')
    if not is_generic(right) and not _is_node(right):
        raise TypeError('right side of >> must be a Node')
    merged = dict(_components(left))
    for name, value in _components(right).items():
        candidate, suffix = name, 2
        while candidate in merged:
            candidate = f'{name}_{suffix}'
            suffix += 1
        merged[candidate] = value
    return serial(**merged)
