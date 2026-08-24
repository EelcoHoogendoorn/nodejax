"""taps: observe every wire of a composite — capture_intermediates as a
transform, riding the ordinary aux stream."""

import pytest
import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Leaf, taps
from nodejax.control import Gain


def test_taps_observes_every_wire():
    pipe = Gain() >> Gain()
    tapped = taps(pipe).parameterize(gain=Struct(scale=2.0), gain_2=Struct(scale=3.0))
    out, wires = tapped.apply(1.0)

    assert out == 6.0
    assert wires.gain == 2.0
    assert wires.gain_2 == 6.0

    with pytest.raises(TypeError, match='serial pipe'):
        taps(Gain())


def test_taps_preserves_serial_entropy_routing():
    def noisy():
        def apply(input, rng):
            return input + jax.random.normal(rng.next())

        return Leaf(apply, name='noisy')

    key = jax.random.PRNGKey(4)
    pipe = noisy() >> noisy()
    expected = pipe.apply(1.0, rng=key)
    actual, wires = taps(pipe).apply(1.0, rng=key)

    assert jnp.allclose(actual, expected)
    assert not jnp.allclose(wires.noisy, wires.noisy_2)
