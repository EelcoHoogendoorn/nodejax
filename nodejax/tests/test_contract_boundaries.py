"""The public and compiled contract boundaries."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Composite, Leaf, Node, Struct, nn, serial
from nodejax.control import Gain, Integrator
from nodejax.rng import MaybeKeyStream


KEY = jax.random.PRNGKey(0)


def _stream(key=None):
    return MaybeKeyStream() if key is None else MaybeKeyStream(key)


def Scale() -> Node:
    def apply(input, scale):
        return input * scale

    return Leaf(apply, name='scale').node


def StatefulScale() -> Node:
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(param, input, offset=0.0):
        return param.scale * input + offset

    def apply(param, state, input):
        next_state = state + param.scale * input
        return next_state, next_state

    return Leaf(apply, param=param, init=init, name='stateful_scale')


def test_multi_field_with_input_accepts_a_formed_bundle():
    node = Scale().with_input(Struct(input=jnp.ones(3)))
    model = node.parameterize()

    output = model.apply(input=jnp.ones(3), scale=3.0)
    assert jnp.allclose(output, 3.0)


def test_multi_field_with_input_rejects_one_bare_wire():
    with pytest.raises(TypeError, match='multi-input call requires a Struct'):
        Scale().with_input(jnp.ones(3))


def test_apply_inputs_cannot_have_defaults():
    def apply(input, scale=2.0):
        return input * scale

    with pytest.raises(TypeError, match="input 'scale' cannot have a default"):
        Leaf(apply, name='defaulted_apply')


def test_closed_explicit_bundle_selects_declared_fields():
    model = Leaf(lambda input: input, name='identity')

    output = model.apply(bundle=Struct(input=1.0, extra=2.0))

    assert output == 1.0
    with pytest.raises(TypeError, match='unknown input fields'):
        model.apply(input=1.0, extra=2.0)
    with pytest.raises(TypeError, match='does not accept rng'):
        model.apply(input=1.0, rng=KEY)
    with pytest.raises(TypeError, match='beside bundle'):
        model.apply(bundle=Struct(input=1.0, rng=KEY))


def test_rng_is_not_reserved_inside_an_ordinary_domain_value():
    model = Leaf(
        lambda input: input.rng + input.value,
        name='domain_rng',
    )
    value = Struct(rng=jnp.asarray(2.0), value=jnp.asarray(3.0))

    assert model.apply(value) == 5.0
    resolved = model.with_input(value)
    assert resolved.apply(value) == 5.0


def test_with_input_distinguishes_a_struct_wire_from_a_formed_bundle():
    node = Leaf(lambda input: input, name='identity').node
    value = Struct(input=jnp.ones(3))

    from_wire = node.with_input(value)
    from_bundle = node.with_input(bundle=value)

    assert from_wire.contract.input_spec.input.input.shape == (3,)
    assert from_bundle.contract.input_spec.input.shape == (3,)
    with pytest.raises(TypeError, match='one input wire or bundle='):
        node.with_input(value, bundle=value)


def test_raw_composite_uses_one_explicit_bundle_contract():
    def apply(param, state, input):
        return state, input.input

    node = Composite()(apply, name='raw')
    bundle = Struct(input=jnp.ones(3))
    declared = Composite()(
        apply, apply_input_spec=bundle, name='declared_raw')

    assert jnp.allclose(node.apply(input=jnp.ones(3)), jnp.ones(3))
    assert jnp.allclose(node.apply(bundle=bundle), jnp.ones(3))
    assert node.with_input(bundle=bundle).contract.input_spec.input.shape == (3,)
    assert declared.contract.input_spec.input.shape == (3,)

    with pytest.raises(TypeError, match='no fixed input fields'):
        node.apply(jnp.ones(3))
    with pytest.raises(TypeError, match='formed call bundle'):
        node.with_input(jnp.ones(3))
    with pytest.raises(TypeError, match='formed call bundle'):
        Composite()(
            apply, apply_input_spec=jnp.ones(3), name='invalid_raw')


def test_authored_wiring_keeps_a_unary_struct_input_whole():
    def param(node):
        return jnp.asarray(node.input.payload.shape[0])

    def init(param, input):
        return input

    def apply_child(param, state, input):
        return state, param * input.payload

    child = Leaf(
        apply_child, param=param, init=init, name='struct_child')

    def apply(self, input):
        return self.child(input)

    input = Struct(payload=jnp.ones(3))
    node = Composite(child=child)(
        apply, name='struct_parent').with_input(input)
    model = node.parameterize().initialize(input=input)

    assert model.param.child == 3
    assert model.state.child.payload.shape == (3,)
    successor, output = model(input)
    assert successor.state.child.payload.shape == (3,)
    assert jnp.allclose(output, 3.0)


def test_zero_input_with_input_rejects_a_wire():
    source = Leaf(
        lambda rng: rng.next(),
        name='random_source',
    ).node

    with pytest.raises(TypeError, match='zero-input call cannot consume a wire'):
        source.with_input(Struct(rng=KEY))


def test_closed_explicit_bundle_keeps_required_rng_and_required_fields():
    def apply(input, rng):
        return input + jax.random.normal(rng.next(), ())

    model = Leaf(apply, name='random')
    bundle = Struct(input=1.0, extra=2.0)

    assert model.apply(bundle=bundle, rng=KEY) == model.apply(
        input=1.0, rng=KEY)
    with pytest.raises(TypeError, match='missing required input fields'):
        model.apply(bundle=Struct(extra=2.0), rng=KEY)
    with pytest.raises(TypeError, match='apply requires rng='):
        model.apply(bundle=bundle)


@pytest.mark.parametrize('formed', [False, True], ids=['arguments', 'formed'])
def test_resolved_value_spec_does_not_narrow_runtime_shape(formed: bool):
    model = Leaf(lambda input: input, name='identity').with_input(jnp.zeros(3))
    runtime_input = jnp.zeros(4)

    output = (model.apply(bundle=Struct(input=runtime_input)) if formed
              else model.apply(runtime_input))
    assert output.shape == (4,)


def test_public_param_and_state_construction_validate_their_bundles():
    with pytest.raises(TypeError, match='unknown'):
        Gain().parameterize(scale=2.0, extra=1.0)

    with pytest.raises(TypeError, match='unknown'):
        Integrator().parameterize().initialize(extra=1.0)


def test_public_calls_enforce_required_rng_metadata():
    def param(rng):
        return Struct(key=rng.next())

    def init(rng):
        return Struct(key=rng.next())

    def apply(param, state, input, rng):
        return state, input + jax.random.normal(rng.next(), ())

    node = Leaf(apply, param=param, init=init, name='random').node

    with pytest.raises(TypeError, match='parameterize requires rng='):
        node.parameterize()

    parameterized = node.parameterize(rng=KEY)
    with pytest.raises(TypeError, match='init requires rng='):
        parameterized.initialize()

    initialized = parameterized.initialize(rng=KEY)
    with pytest.raises(TypeError, match='apply requires rng='):
        initialized.apply(1.0)


def test_canonical_construction_requires_formed_struct_bundles():
    parametric = Gain()
    cyclic = Integrator().node

    with pytest.raises(TypeError, match='formed Struct'):
        parametric.contract.param(
            (), _stream())
    with pytest.raises(TypeError, match='formed Struct'):
        cyclic.contract.init((), (), _stream())


def test_node_contract_exposes_formed_param_init_apply():
    node = Integrator()
    param = node.contract.param(
        Struct(decay=0.0), _stream())
    state = node.contract.init(
        param, Struct(), _stream())
    next_state, output = node.contract.apply(
        param, state, Struct(input=jnp.asarray(2.0)),
        _stream())

    assert jnp.allclose(next_state, 2.0)
    assert jnp.allclose(output, 2.0)


def test_contract_uses_the_node_it_is_read_from():
    base = nn.Linear(2)
    narrow = base.with_input(jnp.zeros(3))
    wide = base.with_input(jnp.zeros(5))

    narrow_param = narrow.contract.param(
        Struct(), _stream(KEY))
    wide_param = wide.contract.param(
        Struct(), _stream(KEY))

    assert narrow_param.w.shape == (3, 2)
    assert wide_param.w.shape == (5, 2)


def test_public_bindings_and_contract_share_operations():
    node = StatefulScale()
    param_input = Struct(scale=2.0)
    state_input = Struct(offset=3.0)
    input = jnp.asarray(4.0)

    parameter_bound = node.parameterize(param_input)
    contract_param = node.contract.param(
        param_input, _stream())
    assert jnp.allclose(parameter_bound.param.scale, contract_param.scale)

    state_bound = parameter_bound.initialize(state_input, input=input)
    contract_state = node.contract.prime(
        contract_param, state_input, input,
        _stream())
    assert jnp.allclose(state_bound.state, contract_state)

    contract_next, contract_output = node.contract.apply(
        contract_param, contract_state, Struct(input=input),
        _stream())
    surface_next, surface_output = parameter_bound.apply(state_bound.state, input)
    assert jnp.allclose(surface_next, contract_next)
    assert jnp.allclose(surface_output, contract_output)


def test_contract_is_the_only_direct_operation_surface():
    node = Integrator()
    duplicated = [
        name for name in
        ('param_fn', 'init_fn', 'apply_fn', 'build_param', 'build_state')
        if name in dir(node)
    ]

    assert duplicated == []


def test_serial_preserves_a_struct_valued_single_wire():
    source = Leaf(lambda input: Struct(input=input), name='source')
    sink = Leaf(lambda input: input, name='sink')

    value = Struct(input=3.0)
    direct = sink.apply(value)
    composed = serial(source=source, sink=sink).apply(3.0)

    assert type(composed) is Struct
    assert composed.__keys__ == direct.__keys__ == value.__keys__
    assert composed.input == direct.input == value.input
