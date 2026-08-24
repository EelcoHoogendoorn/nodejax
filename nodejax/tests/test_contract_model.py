"""The uniform T4 call records and their T3 Contract view."""

import jax.numpy as jnp
import pytest

from nodejax import Leaf, REQUIRED

from nodejax.contract import (
    ApplyCall, CallField, CallForm, Contract, ContractCalls, InitCall,
    ParamCall,
)
from nodejax.definition import Def
from nodejax.node import Node
from nodejax.rng import MaybeKeyStream
from nodejax.struct import Struct


def apply_call(**changes):
    values = dict(
        impl=lambda definition, param, state, input, frame: (state, input),
        form=CallForm.from_values(Struct(input=REQUIRED)),
    )
    values.update(changes)
    return ApplyCall(**values)


def param_call(**changes):
    values = dict(
        impl=lambda definition, input, frame: input,
        form=CallForm.from_values(Struct(width='required')),
    )
    values.update(changes)
    return ParamCall(**values)


def init_call(**changes):
    values = dict(
        impl=lambda definition, param, input, frame: input,
        form=CallForm.from_values(Struct()),
    )
    values.update(changes)
    return InitCall(**values)


def test_call_form_preserves_order_defaults_and_openness():
    declaration = Struct(input='required', scale=2.0)
    form = CallForm.from_values(declaration, open=True)

    assert form.declaration.__keys__ == ('input', 'scale')
    assert form.declaration.scale == 2.0
    assert not form.fields.scale.is_nested
    assert form.open


def test_call_form_projects_complete_and_nested_structs_the_same_way():
    content = Struct(scale=2.0)
    nested = CallForm.from_values(content)
    form = CallForm(Struct(
        complete=CallField.complete(content),
        nested=CallField.nested(nested),
    ))

    assert form.declaration.complete.scale == 2.0
    assert form.declaration.nested.scale == 2.0
    assert not form.fields.complete.is_nested
    assert form.fields.nested.is_nested


def test_parameter_formation_distinguishes_equal_struct_projections():
    content = Struct(scale=2.0)
    form = CallForm(Struct(
        complete=CallField.complete(content),
        nested=CallField.nested(CallForm.from_values(content)),
    ))
    contract = Node(Def(
        'structured',
        ContractCalls(apply=apply_call(), param=param_call(form=form)),
    )).contract

    default = contract.param(Struct(), MaybeKeyStream())
    replaced = contract.param(
        Struct(complete=None, nested=Struct(scale=3.0)),
        MaybeKeyStream(),
    )

    assert default.complete.scale == 2.0
    assert default.nested.scale == 2.0
    assert replaced.complete is None
    assert replaced.nested.scale == 3.0


def test_input_evidence_explicitly_distinguishes_a_wire_from_a_bundle():
    contract = Leaf(lambda input: input).contract
    wire = Struct(value=jnp.zeros(3))

    from_wire = contract.for_input(wire)
    from_bundle = contract._resolve_def(
        Struct(input=wire), bundled=True).contract

    assert from_wire.input_spec.input.value.shape == (3,)
    assert from_bundle.input_spec.input.value.shape == (3,)


def test_call_form_intake_uses_the_declared_structure():
    value = Struct(payload=1.0)
    single = CallForm.from_values(Struct(input=REQUIRED))
    multiple = CallForm.from_values(Struct(left=REQUIRED, right=REQUIRED))
    open_form = CallForm.from_values(Struct(), open=True)

    assert single.intake(Struct(input=value)) is value
    assert multiple.intake(Struct(left=1.0, right=2.0)).left == 1.0
    assert open_form.intake(value) is value


def test_open_call_form_consumes_a_formed_bundle():
    form = CallForm.from_values(Struct(input=REQUIRED), open=True)
    bundle = Struct(input=1.0, side=2.0)

    assert form.feed(bundle) is bundle
    with pytest.raises(TypeError, match='formed call bundle must be a Struct'):
        form.feed(1.0)


def test_open_apply_form_requires_fixed_fields_and_preserves_extras():
    form = CallForm.from_values(Struct(input=REQUIRED), open=True)
    contract = Node(Def(
        'open', ContractCalls(apply=apply_call(form=form)))).contract

    _, output = contract.apply(
        (), (), Struct(input=1.0, side=2.0), MaybeKeyStream())

    assert output.input == 1.0
    assert output.side == 2.0
    with pytest.raises(TypeError, match="missing required field 'input'"):
        contract.apply((), (), Struct(side=2.0), MaybeKeyStream())


def test_constructor_call_forms_cannot_be_open():
    form = CallForm.from_values(Struct(), open=True)

    with pytest.raises(TypeError, match='ParamCall form cannot be open'):
        param_call(form=form)
    with pytest.raises(TypeError, match='InitCall form cannot be open'):
        init_call(form=form)


def test_call_field_rejects_malformed_explicit_structure():
    with pytest.raises(TypeError, match='is_nested must be a bool'):
        CallField(1, None)
    with pytest.raises(TypeError, match='nested CallField must contain'):
        CallField(True, Struct())


def test_role_records_contain_only_role_specific_facts():
    param = param_call(takes_rng=True, reads_def=True)
    init = init_call(takes_rng=True, requires_input=True)
    apply = apply_call(
        input_spec=Struct(input='f32[4]'), takes_rng=True)

    assert param.form.declaration.__keys__ == ('width',)
    assert param.reads_def
    assert init.requires_input
    assert init.takes_rng
    assert apply.input_spec.input == 'f32[4]'


def test_calls_presence_is_the_parametric_and_cyclic_source():
    calls = ContractCalls(
        apply=apply_call(), param=param_call(), init=init_call())
    definition = Def('complete', calls)

    assert definition.parametric
    assert definition.cyclic
    assert not definition.copy(
        calls=calls.copy(param=None)).parametric
    assert not definition.copy(
        calls=calls.copy(init=None)).cyclic


def test_contract_is_a_view_over_the_same_definition():
    calls = ContractCalls(apply=apply_call(), param=param_call())
    definition = Def('viewed', calls)
    node = Node(definition)

    assert node._def.calls is calls
    assert definition.calls is calls
    assert node.contract._def is definition


def test_contract_optional_input_evidence_is_a_t3_operation():
    unresolved = Leaf(lambda input: input).contract

    assert unresolved.input_spec_for('input') is None
    assert unresolved.for_input(None) is unresolved
    resolved = unresolved.for_input(jnp.zeros(3))
    assert resolved.input_spec_for('input').shape == (3,)


@pytest.mark.parametrize(
    ('factory', 'changes', 'message'),
    [
        (apply_call, {'impl': 1}, 'callable'),
        (apply_call, {'form': Struct()}, 'CallForm'),
        (apply_call, {'takes_rng': 'required'}, 'bool'),
        (param_call, {'form': Struct()}, 'CallForm'),
        (init_call, {'requires_input': 1}, 'bool'),
    ],
)
def test_role_records_reject_invalid_parts(factory, changes, message):
    with pytest.raises(TypeError, match=message):
        factory(**changes)


def test_contract_calls_reject_wrong_role_records():
    apply = apply_call()
    with pytest.raises(TypeError, match='ApplyCall'):
        ContractCalls(apply=None)
    with pytest.raises(TypeError, match='ParamCall'):
        ContractCalls(apply=apply, param=Struct())
    with pytest.raises(TypeError, match='InitCall'):
        ContractCalls(apply=apply, init=Struct())
