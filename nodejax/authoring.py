"""Public leaf authoring lowered into complete definitions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, Callable

from nodejax.contract import CallForm, ContractCalls
from nodejax.definition import Def
from nodejax.frozendict import frozendict
from nodejax.lifting import (
    _apply_channels, _check_methods, _compile_apply, _compile_init,
    _compile_param,
)
from nodejax.node import BaseNode, Node, _is_node
from nodejax.pnode import PNode
from nodejax.struct import Struct


_EMPTY_MAPPING = frozendict()


def Leaf(apply: Callable | None = None, *, param=None, init=None,
         name: str | None = None, apply_input_spec: Any = None,
         methods: Mapping[str, Callable] = _EMPTY_MAPPING,
         tags=(), boundary: Mapping = _EMPTY_MAPPING):
    """Lower authored functions into a leaf definition."""
    if apply is None:
        return partial(
            Leaf, param=param, init=init, name=name,
            apply_input_spec=apply_input_spec, methods=methods,
            tags=tags, boundary=boundary,
        )

    apply_call = _compile_apply(apply)
    parametric, cyclic = _apply_channels(apply)
    if parametric != (param is not None):
        expectation = 'requires' if parametric else 'does not use'
        raise TypeError(
            f'{name or apply.__name__}: apply {expectation} a param constructor')
    if cyclic != (init is not None):
        expectation = 'requires' if cyclic else 'does not use'
        raise TypeError(
            f'{name or apply.__name__}: apply {expectation} an initializer')
    if apply_input_spec is not None:
        apply_call = apply_call.copy(
            input_spec=apply_call.form.feed(apply_input_spec))

    definition = Def(
        name=name or apply.__name__,
        calls=ContractCalls(
            apply=apply_call,
            param=_compile_param(param) if parametric else None,
            init=_compile_init(init) if cyclic else None,
        ),
        methods=_check_methods(methods),
        tags=frozenset(tags),
        boundaries=frozendict(boundary),
    )
    return Node(definition) if parametric else PNode(definition, ())


def derive(parent: BaseNode, *, apply: Callable | None = None,
           init: Callable | None = None, param: Callable | None = None,
           name: str | None = None, apply_input_spec: Any = None,
           state_input_spec: Any = None,
           methods: Mapping[str, Callable] = _EMPTY_MAPPING,
           tags=None):
    """Lower selective authored overrides against a leaf definition."""
    if not _is_node(parent):
        raise TypeError('derive expects a Node view')
    if parent._def.members:
        raise TypeError('derive extends leaves; wrap structured nodes')

    apply_call = _compile_apply(apply) if apply is not None else parent._def.calls.apply
    parametric, cyclic = (_apply_channels(apply) if apply is not None
                          else (parent.parametric, parent.cyclic))
    param_call = (_compile_param(param) if param is not None
                  else parent._def.calls.param if parametric else None)
    init_call = (_compile_init(init) if init is not None
                 else parent._def.calls.init if cyclic else None)

    if parametric and param_call is None:
        raise TypeError('derived apply requires parameters; supply param=')
    if cyclic and init_call is None:
        raise TypeError('derived apply requires state; supply init=')
    if apply_input_spec is not None:
        apply_call = apply_call.copy(
            input_spec=apply_call.form.feed(apply_input_spec))
    if state_input_spec is not None:
        if init_call is None:
            raise TypeError('state_input_spec requires an initializer')
        if type(state_input_spec) is Struct and 'rng' in state_input_spec:
            raise TypeError('rng is not state input data')
        init_call = init_call.copy(
            form=CallForm.from_values(state_input_spec))

    definition = Def(
        name=name or parent.name,
        calls=ContractCalls(
            apply=apply_call,
            param=param_call if parametric else None,
            init=init_call if cyclic else None,
        ),
        methods=_check_methods({
            **dict(parent._def.methods),
            **dict(methods),
        }),
        tags=parent.tags if tags is None else frozenset(tags),
        boundaries=parent._def.boundaries,
    )
    return Node(definition) if parametric else PNode(definition, ())
