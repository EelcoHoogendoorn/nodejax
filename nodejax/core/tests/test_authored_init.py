"""An init written against ``self``: member views over the init-time slots.
``bind`` places a state, ``reset`` primes a member from an input the init
computes, a read builds a member from its own bundle, and the composite's
state is what the slots hold when the init returns. Binding the members'
parameters and splitting their keys is the views' job, not the author's.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Composite, Leaf, Node, Wrapper, node
from nodejax.struct import Struct


def Counter() -> Node:
    """State counts up by the input; the output is the count before."""
    return Leaf(
        lambda state, input: (state + input, state),
        init=lambda: jnp.zeros(()),
        name='counter',
    ).node


def Follower() -> Node:
    """State primed from the first input's value, then tracking the input."""
    return Leaf(
        lambda state, input: (input, state),
        init=lambda input: input * 2.0,
        name='follower',
    ).node


def Noisy() -> Node:
    """State drawn at init: the members' keys are split by the views."""
    return Leaf(
        lambda state, input: (state, state + input),
        init=lambda rng: jax.random.normal(rng.next(), ()),
        name='noisy',
    ).node


@node
def Pair() -> Node:
    """A counter placed from the start and a follower primed off it."""
    members = Composite(count=Counter(), follow=Follower())

    def apply(self, input):
        return self.follow(self.count(input))

    def init(self, start):
        self.count.bind(state=start)
        self.follow.reset(input=self.count.state + 1.0)

    return members(apply, init=init)


def test_bind_and_reset_through_self_are_the_composite_state():
    program = Pair().parameterize().initialize(start=jnp.asarray(3.0))
    assert program.state.count == 3.0
    assert program.state.follow == 8.0  # (3 + 1) * 2, primed from what the init computed
    advanced, out = program.apply(jnp.asarray(1.0))
    assert out == 8.0 and advanced.state.count == 4.0 and advanced.state.follow == 3.0


def test_the_init_takes_no_key_when_no_member_draws():
    assert Pair().contract.init_takes_rng is False
    with pytest.raises(TypeError):
        Pair().parameterize().init(rng=jax.random.PRNGKey(0), start=jnp.asarray(0.0))


def test_a_member_that_draws_gets_its_key_from_the_view():
    @node
    def Drawn() -> Node:
        members = Composite(count=Counter(), noise=Noisy())

        def apply(self, input):
            return self.noise(self.count(input))

        def init(self, start):
            self.count.bind(state=start)
            self.noise.reset()

        return members(apply, init=init)

    assert Drawn().contract.init_takes_rng is True
    first = Drawn().parameterize().init(rng=jax.random.PRNGKey(0), start=jnp.asarray(0.0))
    second = Drawn().parameterize().init(rng=jax.random.PRNGKey(1), start=jnp.asarray(0.0))
    assert first.noise != second.noise


def test_an_untouched_member_is_built_from_its_own_bundle():
    @node
    def Partial() -> Node:
        members = Composite(count=Counter(), other=Counter())

        def apply(self, input):
            return self.other(self.count(input))

        def init(self, start):
            self.count.bind(state=start)

        return members(apply, init=init)

    state = Partial().parameterize().init(start=jnp.asarray(5.0))
    assert state.count == 5.0 and state.other == 0.0


def test_a_wrapper_init_written_against_self_primes_its_member():
    @node
    def Primed(follow: Node) -> Node:
        def apply(self, input):
            return self.follow(input)

        def init(self, start):
            self.follow.reset(input=start)

        return Wrapper(follow=follow)(apply, init=init)

    program = Primed(Follower()).parameterize().initialize(start=jnp.asarray(1.5))
    assert program.state == 3.0


def test_an_init_against_self_returns_nothing():
    @node
    def Returning() -> Node:
        members = Composite(count=Counter())

        def apply(self, input):
            return self.count(input)

        def init(self, start):
            self.count.bind(state=start)
            return Struct(count=start)

        return members(apply, init=init)

    with pytest.raises(TypeError, match='returns nothing'):
        Returning().parameterize().init(start=jnp.asarray(0.0))


def test_an_init_against_self_takes_no_param_or_state_argument():
    def apply(self, input):
        return self.count(input)

    def init(self, param, start):
        self.count.bind(state=start)

    with pytest.raises(TypeError, match='not arguments'):
        Composite(count=Counter())(apply, init=init)
