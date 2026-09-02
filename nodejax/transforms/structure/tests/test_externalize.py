"""Tests for moving one member's params into the input."""

import jax
import jax.numpy as jnp

import pytest

from nodejax import Composite, externalize, nn
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_externalize_member():
    pipe = Gain() >> Gain()
    ext = externalize(pipe, 'gain_2')

    node = ext.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                            gain_2=Struct(scale=jnp.asarray(0.0)))
    assert node.param.gain_2 == ()               # externalized: an empty slot

    out = node.apply(gain_2=Struct(scale=jnp.asarray(3.0)), input=1.0)
    assert jnp.allclose(out, 6.0)                # 1 * 2 * 3, world bound from input

    reference = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                                  gain_2=Struct(scale=jnp.asarray(3.0)))
    assert jnp.allclose(out, reference.apply(1.0))


def test_externalize_at_init():
    """A cyclic pipe's init spec-propagates by running each member, so
    the externalized slot needs values there; at_init supplies the
    stand-in, and apply still binds the member from the input."""
    pipe = Gain() >> Integrator()

    bare = externalize(pipe, 'gain').parameterize(
        gain=Struct(scale=jnp.asarray(0.0)))
    with pytest.raises(TypeError):
        bare.with_input(
            Struct(gain=Struct(scale=jnp.asarray(0.0)), input=0.0)
        ).bind(bare.param).init()

    ext = externalize(pipe, 'gain', at_init=Struct(scale=jnp.asarray(1.0)))
    node = ext.parameterize(gain=Struct(scale=jnp.asarray(0.0)))
    state = node.with_input(
        Struct(gain=Struct(scale=jnp.asarray(0.0)), input=0.0)
    ).bind(node.param).init()
    _, out = node.apply(state, gain=Struct(scale=jnp.asarray(3.0)), input=2.0)
    assert jnp.allclose(out, 6.0)                # 0 + (2 * 3), accumulated once


def test_externalize_composite_subtree() -> None:
    critic = (
        nn.Linear(4) >> nn.tanh >> nn.Linear(1)
    ).with_input(jnp.zeros(3))
    members = Composite(critic=critic)

    def apply(self, input):
        return self.critic(input)

    model = members(apply, name='model').with_input(jnp.zeros(3))
    externalized = externalize(model, 'critic')
    bound = externalized.parameterize(rng=jax.random.PRNGKey(0))
    critic_param = critic.parameterize(rng=jax.random.PRNGKey(1)).param
    input = jnp.arange(3.0)

    assert bound.param.critic == ()
    assert jnp.allclose(
        bound.apply(critic=critic_param, input=input),
        model.bind(Struct(critic=critic_param)).apply(input),
    )


def test_externalize_whole_node() -> None:
    critic = (nn.Linear(4) >> nn.tanh >> nn.Linear(1)).with_input(jnp.zeros(3))
    external = externalize(critic, field='critic')
    weights = critic.parameterize(rng=jax.random.PRNGKey(1)).param
    input = jnp.arange(3.0)

    assert not external.parametric
    bound = external.parameterize()
    assert jnp.allclose(
        bound.apply(input=input, critic=weights),
        critic.bind(weights).apply(input),
    )


def test_externalized_call_stays_flat_beside_the_node_fields() -> None:
    """A node with several call fields keeps them; the parameter field joins."""
    members = Composite(gain=Gain(), offset=Gain())

    def apply(self, input, shift):
        return self.gain(input) + self.offset(shift)

    model = members(apply, name='model').with_input(Struct(input=0.0, shift=0.0))
    external = externalize(model, 'gain').parameterize(
        gain=Struct(scale=jnp.asarray(0.0)), offset=Struct(scale=jnp.asarray(2.0)))
    out = external.apply(input=3.0, shift=1.0, gain=Struct(scale=jnp.asarray(5.0)))

    assert external.contract.apply_fields == ('input', 'shift', 'gain')
    assert jnp.allclose(out, 3.0 * 5.0 + 1.0 * 2.0)
