"""drop_aux stops Aux at a Node boundary or runtime value."""

import jax
import jax.numpy as jnp

from nodejax import Aux, Leaf, Node, drop_aux, node, serial, split_aux
from nodejax.struct import Struct


def Sower() -> Node:
    """A node that sows alongside its output."""
    def apply(param, input):
        return param.w * input, Aux(activity=input ** 2)
    return Leaf(apply, param=lambda w: Struct(w=jnp.asarray(w)), name='sower')


def test_the_clean_signal_alone_comes_out():
    node = Sower().parameterize(w=2.0)
    clean, aux = split_aux(node.apply(3.0))
    assert clean == 6.0 and aux is not None          # it does sow

    dropped = drop_aux(node)
    out = dropped.apply(3.0)
    assert out == 6.0
    value, aux = split_aux(out)
    assert aux is None                               # and now it does not


def test_only_the_nodes_own_emission_is_dropped():
    """A member sowing inside a composite is diverted by the wiring as
    always; what this drops is the composite's re-emission. So dropping at
    the top silences the pipe without changing how members behave."""
    pipe = serial(a=Sower(), b=Sower())
    node = pipe.parameterize(a=Struct(w=2.0), b=Struct(w=3.0))

    clean, aux = split_aux(node.apply(1.0))
    assert clean == 6.0
    assert set(aux.__keys__) == {'a', 'b'}           # both members' sowings

    quiet = drop_aux(node)
    assert quiet.apply(1.0) == 6.0                   # same signal
    value, aux = split_aux(quiet.apply(1.0))
    assert aux is None                               # no Aux remains


def test_a_node_that_sows_nothing_is_unchanged():
    plain = Leaf(lambda input: input * 2.0, name='plain')
    assert drop_aux(plain).apply(3.0) == 6.0


def test_runtime_output_is_cleaned_directly():
    output = 3.0, Aux(activity=4.0)
    assert drop_aux(output) == 3.0
    plain = Struct(left=1.0, right=2.0)
    assert drop_aux(plain) is plain


def test_generic_node_remains_a_transformable_node():
    @node
    def Scaled(factor: float) -> Node:
        return Leaf(lambda input: factor * input)

    deferred = drop_aux(Scaled())
    assert deferred.generic
    assert deferred.name == 'drop_aux'

    complete = deferred.specialize(**{'inner.factor': 3.0}).parameterize()
    assert complete.apply(2.0) == 6.0
