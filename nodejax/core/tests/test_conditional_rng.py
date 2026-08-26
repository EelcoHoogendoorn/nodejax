"""Conditional RNG forwarding keeps authored code uniform and APIs exact."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import KeyStream, Leaf, Wrapper
from nodejax.transforms.transform import MaybeKeyStream


def identity():
    return Leaf(lambda input: input, name='identity')


def noise():
    def apply(input, rng):
        return input + jax.random.normal(rng.next())

    return Leaf(apply, name='noise')


def forward(target, plan=None):
    anchor = identity()
    source = target.contract.apply_takes_rng if plan is None else plan

    def apply(self, input, rng):
        assert type(rng) is MaybeKeyStream
        self.anchor(input)
        return target(input, rng=rng)

    return Wrapper(anchor=anchor)(
        apply, name='forward', rng_from=source)


def test_empty_forwarding_stream_omits_rng_at_a_deterministic_callee():
    wrapped = forward(identity())

    assert not wrapped.contract.apply_takes_rng
    assert wrapped.apply(2.0) == 2.0
    with pytest.raises(TypeError, match='does not accept rng'):
        wrapped.apply(2.0, rng=jax.random.key(0))


def test_real_forwarding_stream_supplies_rng_to_a_stochastic_callee():
    wrapped = forward(noise())
    key = jax.random.key(3)

    assert wrapped.contract.apply_takes_rng
    with pytest.raises(TypeError, match='requires rng'):
        wrapped.apply(2.0)
    assert jnp.allclose(wrapped.apply(2.0, rng=key),
                        wrapped.apply(2.0, rng=key))


def test_an_empty_forwarding_capability_cannot_feed_a_stochastic_callee():
    empty_to_random = forward(noise(), False)
    with pytest.raises(TypeError, match='random key required'):
        empty_to_random.apply(1.0)


def test_a_keyed_authoring_capability_can_call_a_deterministic_callee():
    key_to_deterministic = forward(identity(), True)
    assert key_to_deterministic.apply(
        1.0, rng=jax.random.key(0)) == 1.0


def test_empty_rng_token_cannot_escape_as_data():
    anchor = identity()

    def apply(self, input, rng):
        self.anchor(input)
        return rng

    leaked = Wrapper(anchor=anchor)(
        apply, name='leaked_rng', rng_from=False)
    with pytest.raises(TypeError, match='KeyStream escaped'):
        leaked.apply(1.0)


def test_rng_from_requires_an_authored_rng_channel():
    with pytest.raises(TypeError, match='requires an rng parameter'):
        Wrapper(anchor=identity())(
            lambda self, input: self.anchor(input),
            rng_from=False)


def test_ordinary_authored_wrapper_gets_the_narrow_stream():
    def apply(self, input, rng):
        assert type(rng) is KeyStream
        return self.anchor(input) + jax.random.normal(rng.next())

    wrapped = Wrapper(anchor=identity())(apply, name='ordinary_rng')
    key = jax.random.key(5)

    assert jnp.allclose(wrapped.apply(1.0, rng=key),
                        wrapped.apply(1.0, rng=key))
