"""taps: observe every wire of a composite — capture_intermediates as a
transform, riding the ordinary aux channel."""

import pytest

from nodejax.struct import Struct
from nodejax import taps
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
