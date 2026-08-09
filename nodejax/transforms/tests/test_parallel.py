"""parallel: named nodes over the strands of a Struct signal."""

import jax.numpy as jnp
import pytest

from nodejax import parallel, node_def
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator
ident = node_def(lambda input: input, name='ident')


def test_parallel_routes_strands():
    block = parallel(x=Gain(), y=ident).parameterize(x=Struct(scale=jnp.asarray(2.0)))
    out = block.apply(Struct(x=jnp.asarray(3.0), y=jnp.asarray(7.0)))
    assert jnp.allclose(out.x, 6.0)              # strand through its member
    assert jnp.allclose(out.y, 7.0)              # explicit identity strand


def test_parallel_cyclic_state_threads():
    block = parallel(x=Integrator(), y=ident).parameterize(x=Struct(gain=jnp.asarray(1.0)))
    state = block.init()
    state, out = block.apply(state, Struct(x=jnp.asarray(2.0), y=jnp.asarray(9.0)))
    state, out2 = block.apply(state, Struct(x=jnp.asarray(2.0), y=jnp.asarray(9.0)))
    assert jnp.allclose(out.x, 2.0) and jnp.allclose(out2.x, 4.0)   # state accumulates
    assert jnp.allclose(out2.y, 9.0)


def test_parallel_is_strict():
    """Every strand is written down: an input field without a member is
    an error, never a silent passthrough."""
    block = parallel(x=Gain()).parameterize(x=Struct(scale=jnp.asarray(2.0)))
    with pytest.raises(TypeError):
        block.apply(Struct(x=jnp.asarray(1.0), y=jnp.asarray(1.0)))


def test_parallel_init_splits_rng_to_stochastic_members():
    """A member whose init consumes entropy gets its own split key; a
    deterministic strand alongside is untouched. Regression: parallel init
    called _needs_rng with the wrong arity, so this path crashed the moment
    an rng met a stochastic member (no test exercised it)."""
    import jax

    def Noisy():
        def init(rng):
            return jax.random.normal(rng.next(), ())
        def apply(state, input):
            return state, input + state
        return node_def(apply, init=init, name='noisy')

    block = parallel(x=Noisy(), y=Integrator()).parameterize(
        y=Struct(gain=jnp.asarray(1.0)))
    state = block.init(rng=jax.random.PRNGKey(0))
    assert state.x.shape == ()            # stochastic member keyed from a split
    assert jnp.allclose(state.y, 0.0)     # deterministic strand untouched
