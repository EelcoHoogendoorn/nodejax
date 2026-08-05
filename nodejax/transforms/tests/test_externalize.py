"""externalize: one member's params move from the tree to the input."""

import jax.numpy as jnp

import pytest

from nodejax import externalize
from nodejax.struct import Struct
from nodejax.examples import gain_def, integrator_def


def test_externalize_member():
    pipe = gain_def() >> gain_def()
    ext = externalize(pipe, 'gain_2')

    node = ext.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                            gain_2=Struct(scale=jnp.asarray(0.0)))
    assert node.param.gain_2 == ()               # externalized: an empty slot

    out = node.apply(Struct(gain_2=Struct(scale=jnp.asarray(3.0)), input=1.0))
    assert jnp.allclose(out, 6.0)                # 1 * 2 * 3, world bound from input

    reference = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                                  gain_2=Struct(scale=jnp.asarray(3.0)))
    assert jnp.allclose(out, reference.apply(1.0))


def test_externalize_at_init():
    """A cyclic pipe's init spec-propagates by running each member, so
    the externalized slot needs values there; at_init supplies the
    stand-in, and apply still binds the member from the input."""
    pipe = gain_def() >> integrator_def()

    bare = externalize(pipe, 'gain').parameterize(
        gain=Struct(scale=jnp.asarray(0.0)), integrator=Struct(gain=jnp.asarray(0.5)))
    with pytest.raises(TypeError):
        bare.with_input(Struct(gain=Struct(scale=jnp.asarray(0.0)), input=0.0)).init()

    ext = externalize(pipe, 'gain', at_init=Struct(scale=jnp.asarray(1.0)))
    node = ext.parameterize(gain=Struct(scale=jnp.asarray(0.0)),
                            integrator=Struct(gain=jnp.asarray(0.5)))
    state = node.with_input(Struct(gain=Struct(scale=jnp.asarray(0.0)), input=0.0)).init()
    _, out = node.apply(state, Struct(gain=Struct(scale=jnp.asarray(3.0)), input=2.0))
    assert jnp.allclose(out, 3.0)                # 0 + 0.5 * (2 * 3)