"""Public leaf authoring lowered into complete definitions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, Callable

from nodejax.core.contract import (
    ApplyCall, CallForm, ContractCalls, InitCall, ParamCall,
)
from nodejax.core.definition import Def
from nodejax.frozendict import frozendict
from nodejax.core.lifting import (
    _check_methods, _compile_apply, _compile_init, _compile_param,
)
from nodejax.core.node import BaseNode, Node, _is_node
from nodejax.core.pnode import PNode
from nodejax.struct import Struct
from nodejax.core.types import PyTree


_EMPTY_MAPPING = frozendict()


def _combined_form(parent: CallForm, child: CallForm,
                   role: str) -> CallForm:
    """Combine two constructor inputs without guessing their structure."""
    collisions = set(parent.fields.__keys__) & set(child.fields.__keys__)
    if collisions:
        raise TypeError(
            f'derive {role} constructor inputs overlap: '
            f'{sorted(collisions)}')
    return CallForm(Struct(**{
        **dict(parent.fields.__items__),
        **dict(child.fields.__items__),
    }))


def _constructor_input(form: CallForm, formed_input: Struct) -> Struct:
    """Project a combined constructor input onto one declared call form."""
    return Struct(**{
        name: formed_input[name] for name in form.fields.__keys__
    })


def _merged_struct(parent: Struct, child: Struct, role: str) -> Struct:
    """Shallowly merge two disjoint constructor result fragments."""
    if (not issubclass(type(parent), Struct)
            or not issubclass(type(child), Struct)):
        raise TypeError(
            f'derive can merge {role} only when both constructors return '
            'Struct values')
    collisions = set(parent.__keys__) & set(child.__keys__)
    if collisions:
        raise TypeError(
            f'derive {role} fields overlap: {sorted(collisions)}')
    return Struct(**{
        **dict(parent.__items__),
        **dict(child.__items__),
    })


def _merge_param_calls(parent: ParamCall, child: ParamCall) -> ParamCall:
    """Lower two parameter constructors into one canonical call."""
    form = _combined_form(parent.form, child.form, 'param')

    def impl(definition, formed_input, rng):
        parent_param = parent.impl(
            definition,
            _constructor_input(parent.form, formed_input),
            rng.child(parent.takes_rng),
        )
        child_param = child.impl(
            definition,
            _constructor_input(child.form, formed_input),
            rng.child(child.takes_rng),
        )
        return _merged_struct(parent_param, child_param, 'param')

    return parent.copy(
        impl=impl,
        form=form,
        takes_rng=parent.takes_rng or child.takes_rng,
        reads_def=parent.reads_def or child.reads_def,
    )


def _run_init(call: InitCall, definition: Def, param,
              formed_input: Struct, rng) -> PyTree:
    """Invoke one non-priming initializer with its declared inputs."""
    values = _constructor_input(call.form, formed_input)
    return call.impl(
        definition, param, values, rng.child(call.takes_rng))


def _run_prime(call: InitCall, definition: Def, param,
               formed_input: Struct, input, rng) -> PyTree:
    """Invoke one initializer while a real input is available."""
    values = _constructor_input(call.form, formed_input)
    child_rng = rng.child(call.takes_rng)
    if call.requires_input:
        return call.impl(definition, param, values, input, child_rng)
    return call.impl(definition, param, values, child_rng)


def _merge_init_calls(parent: InitCall, child: InitCall) -> InitCall:
    """Lower two state constructors into one canonical call."""
    form = _combined_form(parent.form, child.form, 'state')
    requires_input = parent.requires_input or child.requires_input

    if requires_input:
        def impl(definition, param, formed_input, input, rng):
            parent_state = _run_prime(
                parent, definition, param, formed_input, input, rng)
            child_state = _run_prime(
                child, definition, param, formed_input, input, rng)
            return _merged_struct(parent_state, child_state, 'state')
    else:
        def impl(definition, param, formed_input, rng):
            parent_state = _run_init(
                parent, definition, param, formed_input, rng)
            child_state = _run_init(
                child, definition, param, formed_input, rng)
            return _merged_struct(parent_state, child_state, 'state')

    return parent.copy(
        impl=impl,
        form=form,
        takes_rng=parent.takes_rng or child.takes_rng,
        requires_input=requires_input,
        reads_def=parent.reads_def or child.reads_def,
    )


def _preserve_derived_state(call: ApplyCall) -> ApplyCall:
    """Overlay an inherited transition onto the complete derived state."""
    def impl(definition, param, state, formed_input, rng):
        if not issubclass(type(state), Struct):
            raise TypeError('an inherited transition requires Struct state')
        next_state, output = call.impl(
            definition, param, state, formed_input, rng)
        if not issubclass(type(next_state), Struct):
            raise TypeError(
                'an inherited transition must return a Struct state fragment')
        unknown = set(next_state.__keys__) - set(state.__keys__)
        if unknown:
            raise TypeError(
                f'an inherited transition returned unknown state fields: '
                f'{sorted(unknown)}')
        return state.replace(**dict(next_state.__items__)), output

    return call.copy(impl=impl)


def _complete_derived_state(call: ApplyCall) -> ApplyCall:
    """Require an overridden transition to return the complete state."""
    def impl(definition, param, state, formed_input, rng):
        next_state, output = call.impl(
            definition, param, state, formed_input, rng)
        if (not issubclass(type(state), Struct)
                or not issubclass(type(next_state), Struct)):
            raise TypeError(
                'an overridden transition over merged state must return '
                'a complete Struct')
        expected = set(state.__keys__)
        actual = set(next_state.__keys__)
        if actual != expected:
            raise TypeError(
                'an overridden transition must return every state field: '
                f'expected {sorted(expected)}, got {sorted(actual)}')
        return state.replace(**dict(next_state.__items__)), output

    return call.copy(impl=impl)


def Leaf(apply: Callable | None = None, *, param=None, init=None,
         name: str | None = None, apply_input_spec: Any = None,
         methods: Mapping[str, Callable] = _EMPTY_MAPPING,
         tags=(), boundary: Mapping = _EMPTY_MAPPING):
    """Lower authored functions; constructor presence defines owned roles."""
    if apply is None:
        return partial(
            Leaf, param=param, init=init, name=name,
            apply_input_spec=apply_input_spec, methods=methods,
            tags=tags, boundary=boundary,
        )

    definition_name = name or apply.__name__
    param_call = _compile_param(param) if param is not None else None
    init_call = (_compile_init(
        init, owner=definition_name, allow_self=False)
                 if init is not None else None)
    apply_call = _compile_apply(
        apply,
        parametric=param_call is not None,
        cyclic=init_call is not None,
        owner=definition_name,
    )
    if apply_input_spec is not None:
        apply_call = apply_call.copy(
            input_spec=apply_call.form.feed(apply_input_spec))

    definition = Def(
        name=definition_name,
        calls=ContractCalls(
            apply=apply_call,
            param=param_call,
            init=init_call,
        ),
        methods=_check_methods(methods),
        tags=frozenset(tags),
        boundaries=frozendict(boundary),
    )
    return Node(definition) if param_call is not None else PNode(definition, ())


def derive(parent: BaseNode, *, apply: Callable | None = None,
           init: Callable | None = None, param: Callable | None = None,
           name: str | None = None, apply_input_spec: Any = None,
           methods: Mapping[str, Callable] = _EMPTY_MAPPING,
           tags=None):
    """Extend a leaf with disjoint Struct constructor fragments.

    An omitted constructor is inherited. If both parent and child define one
    role, their input names and returned Struct fields must be disjoint. Apply
    and methods may replace inherited behavior; tags form a union.
    """
    if not _is_node(parent):
        raise TypeError('derive expects a Node view')
    if parent.state_bound or (parent.parametric and parent.bound):
        raise TypeError('derive expects an unbound Node view')
    if parent._def.members:
        raise TypeError('derive extends leaves; wrap structured nodes')

    definition_name = name or parent.name
    parent_param = parent._def.calls.param
    child_param = _compile_param(param) if param is not None else None
    if parent_param is None:
        param_call = child_param
    elif child_param is None:
        param_call = parent_param
    else:
        param_call = _merge_param_calls(parent_param, child_param)

    parent_init = parent._def.calls.init
    child_init = (_compile_init(
        init, owner=definition_name, allow_self=False)
                  if init is not None else None)
    if parent_init is None:
        init_call = child_init
    elif child_init is None:
        init_call = parent_init
    else:
        init_call = _merge_init_calls(parent_init, child_init)

    apply_call = (_compile_apply(
        apply,
        parametric=param_call is not None,
        cyclic=init_call is not None,
        owner=definition_name,
    ) if apply is not None else parent._def.calls.apply)
    if parent_init is not None and child_init is not None:
        apply_call = (_preserve_derived_state(apply_call)
                      if apply is None else
                      _complete_derived_state(apply_call))
    if apply_input_spec is not None:
        apply_call = apply_call.copy(
            input_spec=apply_call.form.feed(apply_input_spec))

    definition = Def(
        name=definition_name,
        calls=ContractCalls(
            apply=apply_call,
            param=param_call,
            init=init_call,
        ),
        methods=_check_methods({
            **dict(parent._def.methods),
            **dict(methods),
        }),
        tags=parent.tags | (frozenset() if tags is None else frozenset(tags)),
        boundaries=parent._def.boundaries,
    )
    return Node(definition) if param_call is not None else PNode(definition, ())
