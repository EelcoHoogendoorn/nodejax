"""A transform authored entirely through the supported Tier-3 surface."""

import jax.numpy as jnp
import pytest

import nodejax as nx
import nodejax.transform as tx


@tx.transform(preserves='param')
def negate(inner: nx.Node) -> nx.Node:
    def apply(contract, param, state, input, rng):
        assert type(rng) is tx.MaybeKeyStream
        current = contract.members.inner
        state, output = current.apply(
            param, state, input, rng)
        return state, -output

    return nx.Wrapper(inner=inner).roles(
        name=f'negate({inner.name})',
        apply=apply,
    )


def test_public_transform_surface_builds_and_preserves_parameters():
    gain = nx.control.Gain().parameterize(scale=2.0)
    transformed = negate(gain)

    assert type(transformed) is nx.PNode
    assert transformed.param is gain.param
    assert jnp.allclose(transformed.apply(3.0), -6.0)


def test_parameter_preserving_transform_refuses_bound_state():
    model = nx.control.Integrator().parameterize().initialize()

    with pytest.raises(TypeError, match='does not preserve state'):
        negate(model)


def test_layout_changing_transform_refuses_bound_parameters():
    @tx.transform
    def bare_only(inner):
        return negate(inner)

    bound = nx.control.Gain().parameterize(scale=2.0)
    with pytest.raises(TypeError, match='does not preserve parameters'):
        bare_only(bound)


def test_transform_metadata_types_are_available_without_private_imports():
    assert not tx.MaybeKeyStream()
    assert callable(tx.MaybeKeyStream.axis)
    with pytest.raises(AttributeError):
        nx.MaybeKeyStream
    with pytest.raises(AttributeError):
        nx.KeyStream(jnp.asarray([0, 0], dtype='uint32')).axis
    assert tx.AxisSpec(jnp.zeros(())).count is None


def test_t4_call_records_do_not_leak_into_transform_authoring():
    contract = nx.control.Gain().contract

    with pytest.raises(AttributeError):
        contract.apply_form
    with pytest.raises(AttributeError):
        contract.roles
    with pytest.raises(AttributeError):
        tx.CallForm
    with pytest.raises(AttributeError):
        nx.ContractCalls


def test_t3_can_return_a_public_bound_view():
    model = nx.control.Integrator().parameterize().initialize()
    rebound = tx.bind(model.contract, model.param, state=model.state)

    assert type(rebound) is nx.PSNode
    assert rebound.param == model.param
    assert rebound.state == model.state


def test_transparent_wrapper_preserves_struct_constructor_default():
    default = nx.Struct(scale=2.0)

    def param(config=default):
        return config

    def apply(param, input):
        return param.scale * input

    inner = nx.Leaf(apply, param=param)
    wrapped = nx.Wrapper(inner=inner)()

    assert wrapped.parameterize().apply(3.0) == 6.0


def test_authored_wrapper_preserves_resolved_member_input():
    """Wrapper input does not replace a member's own resolved input."""
    def param(scale):
        return nx.Struct(scale=jnp.asarray(scale))

    def init():
        return jnp.zeros(())

    def apply(param, state):
        return state + 1, param.scale

    source = nx.Leaf(apply, param=param, init=init, name='source')

    def shifted(self, offset):
        return self.source() + offset

    wrapped = nx.Wrapper(source=source)(shifted, name='shifted')
    bound = wrapped.with_input(jnp.asarray(3.0)).parameterize(
        scale=jnp.asarray(2.0)).initialize()
    successor, output = bound.apply(jnp.asarray(3.0))

    assert output == 5.0
    assert successor.state == 1.0
