"""sum_junction: one signal in, the members' contributions added.

The fan-out twin of parallel. parallel splits a Struct's strands and keeps
them apart; this broadcasts one signal to every member and sums what comes
back, which is what a block diagram's summing junction does and what an
additive disturbance actually is.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import sum_junction, residual, Leaf, scan, scanned, serial
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def Primed():
    def init(input):
        return jnp.asarray(input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='primed').node


def Streamy():
    def init(rng):
        return Struct(rng=rng)

    def apply(state, input):
        return state, input

    return Leaf(apply, init=init, name='streamy').node


def test_it_adds_what_the_members_make_of_one_input():
    """Every member sees the SAME signal, and the outputs are summed. Not
    the input itself: a junction adds its members, nothing implicit."""
    block = sum_junction(a=Gain(), b=Gain()).parameterize(
        a=Struct(scale=2.0), b=Struct(scale=3.0))

    assert jnp.allclose(block.apply(1.0), 5.0)       # 2 + 3, not 1 + 2 + 3


def test_the_input_is_a_term_only_when_residual_says_so():
    """The two combinators keep their own shapes: residual is x + f(x),
    sum_junction is the f that has several branches."""
    branches = sum_junction(a=Gain(), b=Gain())
    given = dict(a=Struct(scale=2.0), b=Struct(scale=3.0))

    assert jnp.allclose(residual(branches).parameterize(**given).apply(1.0), 6.0)
    assert jnp.allclose(branches.parameterize(**given).apply(1.0), 5.0)


def test_param_and_state_are_keyed_by_member():
    """Composes like every other block: member-keyed Structs, and a
    member's state threads independently of its siblings."""
    block = sum_junction(fast=Integrator(), slow=Integrator()).parameterize(
        fast=Struct(decay=0.0), slow=Struct(decay=0.5))

    state = block.init()
    assert set(state.__keys__) == {'fast', 'slow'}

    state, out = block.apply(state, 1.0)
    state, out = block.apply(state, 1.0)
    assert jnp.allclose(state.fast, 2.0)             # pure accumulation
    assert jnp.allclose(state.slow, 1.5)             # leaks half of what it holds
    assert jnp.allclose(out, 3.5)                    # the sum is what comes out


def test_stochastic_members_get_independent_keys():
    """One boundary key at init routes a split per stochastic member, as
    it does through any composite: two draws, not one shared."""
    def Noise():
        def init(param, rng):
            return Struct(rng=rng)

        def apply(state, input):
            return state, jax.random.normal(state.rng, jnp.shape(input))

        return Leaf(apply, init=init, name='noise')

    block = sum_junction(a=Noise(), b=Noise()).parameterize()
    state = block.with_input(jnp.zeros(())).bind(block.param).init(
        rng=jax.random.PRNGKey(0))

    assert jnp.any(state.a.rng != state.b.rng)


def test_it_is_a_node_like_any_other():
    """Cyclic because a member is, and it scans."""
    block = sum_junction(x=Integrator(), y=Gain())
    assert block.cyclic

    node = scanned(block).parameterize(x=Struct(decay=0.0), y=Struct(scale=10.0))
    outs = node.apply(jnp.array([1.0, 1.0, 1.0]))
    # the integral runs 1, 2, 3 while the gain contributes 10 each step
    assert jnp.allclose(outs, jnp.array([11.0, 12.0, 13.0]))


def test_empty_is_refused():
    with pytest.raises(TypeError, match='at least one member'):
        sum_junction()


def test_a_bound_parametric_member_is_a_transport_container():
    """What `_over_members` bought: the three combinators now settle their
    members the same way. A bound parametric member used to be refused here
    and accepted by a pipe, for no reason either docstring gave. Its params
    are stored construction values, filling the slot that kwargs leave open."""
    block = sum_junction(a=Gain()(scale=2.0), b=Gain())

    assert not block.bound                           # one member short of bound
    assert jnp.allclose(block.parameterize(b=Struct(scale=3.0)).apply(1.0), 5.0)
    # and the same members through a pipe, which always allowed this
    assert jnp.allclose(
        serial(a=Gain()(scale=2.0), b=Gain()).parameterize(b=Struct(scale=3.0)).apply(1.0),
        6.0)


def test_a_generic_member_defers_the_whole_block():
    """The spectrum at the junction's door: any generic member (a
    partial @node call) makes the block generic, and specializing it is
    the same as specializing the members and building afterwards."""
    from nodejax import node as node_annotation

    @node_annotation
    def Scaled(factor):
        return Gain()(scale=float(factor)).node

    block = sum_junction(a=Scaled(), b=Scaled())
    assert block.generic

    built = block.specialize(a=dict(factor=2), b=dict(factor=3))
    assert jnp.allclose(built.parameterize(a=Struct(scale=2.0),
                                           b=Struct(scale=3.0)).apply(1.0), 5.0)


def test_state_bound_primer_is_a_finished_state_slot():
    stored = Primed().bind(state=jnp.asarray(7.0))
    block = sum_junction(p=stored, g=Gain()).parameterize(
        g=Struct(scale=1.0))

    assert not block.contract.init_requires_input
    assert jnp.allclose(block.init().p, 7.0)
    assert jnp.allclose(block.init(p=jnp.asarray(2.0)).p, 2.0)


def test_stored_junction_rng_is_rekeyed():
    original = jax.random.PRNGKey(7)
    stored = Streamy().bind(state=Struct(rng=original))
    block = sum_junction(s=stored, g=Gain()).parameterize(
        g=Struct(scale=1.0))

    with pytest.raises(TypeError, match='rng'):
        block.init()
    state = block.init(rng=jax.random.PRNGKey(1))
    assert not jnp.array_equal(state.s.rng, original)
