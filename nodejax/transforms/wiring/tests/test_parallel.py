"""Verifies `parallel` routing named member Nodes across matching fields of a `Struct` input."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import parallel, Leaf
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator
ident = Leaf(lambda input: input, name='ident')


def Primed():
    def init(input):
        return jnp.asarray(input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='primed').node


def test_parallel_routes_strands():
    block = parallel(x=Gain(), y=ident).parameterize(x=Struct(scale=jnp.asarray(2.0)))
    out = block.apply(x=jnp.asarray(3.0), y=jnp.asarray(7.0))
    assert jnp.allclose(out.x, 6.0)
    assert jnp.allclose(out.y, 7.0)


def test_parallel_cyclic_state_threads():
    block = parallel(x=Integrator(), y=ident).parameterize()
    state = block.init()
    state, out = block.apply(state, x=jnp.asarray(2.0), y=jnp.asarray(9.0))
    state, out2 = block.apply(state, x=jnp.asarray(2.0), y=jnp.asarray(9.0))
    assert jnp.allclose(out.x, 2.0) and jnp.allclose(out2.x, 4.0)
    assert jnp.allclose(out2.y, 9.0)


def test_parallel_is_strict():
    """Verifies that `parallel` rejects input fields that have no matching declared member."""
    block = parallel(x=Gain()).parameterize(x=Struct(scale=jnp.asarray(2.0)))
    with pytest.raises(TypeError):
        block.apply(x=jnp.asarray(1.0), y=jnp.asarray(1.0))


def test_parallel_init_splits_rng_to_stochastic_members():
    """Verifies that `parallel` splits `init` PRNG keys to stochastic members while leaving deterministic siblings untouched."""
    def Noisy():
        def init(rng):
            return jax.random.normal(rng.next(), ())
        def apply(state, input):
            return state, input + state
        return Leaf(apply, init=init, name='noisy')

    block = parallel(x=Noisy(), y=Integrator()).parameterize()
    state = block.init(rng=jax.random.PRNGKey(0))
    assert state.x.shape == ()
    assert jnp.allclose(state.y, 0.0)


def test_state_bound_primer_is_a_finished_state_slot():
    stored = Primed().bind(state=jnp.asarray(7.0))
    block = parallel(p=stored, g=Gain()).parameterize(
        g=Struct(scale=1.0))

    assert not block.contract.init_requires_input
    assert jnp.allclose(block.init().p, 7.0)
    assert jnp.allclose(block.init(p=jnp.asarray(2.0)).p, 2.0)
