"""drop_aux removes Aux from a runtime output."""

from nodejax import Aux, drop_aux
from nodejax.struct import Struct


def test_runtime_output_with_aux_is_cleaned():
    output = 3.0, Aux(activity=4.0)
    assert drop_aux(output) == 3.0


def test_plain_output_without_aux_is_unchanged():
    plain = Struct(left=1.0, right=2.0)
    assert drop_aux(plain) is plain
    assert drop_aux(5.0) == 5.0
