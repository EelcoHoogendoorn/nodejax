"""A shape is not a value: what a transform may do with `node.input`.

`node.input` is zeros of the resolved input spec, and it exists so a walk can
propagate SHAPES through a composition: each member's def resolved against what
the member before it produces. It is not data. `core.Node.input` says so:

    Zero arrays produced by `node.input` must NEVER be passed internally into
    channels or arguments expecting real numerical data (`input`).

The distinction is invisible for almost every node, because almost every init
either ignores its input or reads only its shape. It matters for the ones that
PRIME: a node whose state starts AT its first real sample, where a zero is a
wrong answer rather than a missing one. A derivative's previous-sample register
primed from zeros produces a spike on the first step; a warm filter primed from
zeros is a cold filter.

Such a node declares `init_requires_input`, and when all a walk has is a shape,
the right answer is to REFUSE. `serial` states the rule in its own comment:

    a REAL value threads as a value; a bound shape only resolves each member's
    def. A member that PRIMES from its input refuses when all we have is a
    shape, rather than warm-starting from zeros.

Every transform that walks has to keep that distinction. This file checks that
they do, once each, because the failure is silent: a model that should have
refused instead trains from a subtly wrong start.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import (Node, Leaf, serial, Composite, at, batch, stack, ensemble,
                     repeat, sum_junction, parallel, residual, remat,
                     state_reinit)
from nodejax.struct import Struct

X = jnp.array([3.0, 4.0, 5.0])


def Primed() -> Node:
    """Its state IS its first real sample. Zeros are not a lesser start here,
    they are a different node."""
    def init(input):
        return jnp.asarray(input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='primed').node


def test_the_node_declares_that_it_primes():
    assert Primed().contract.init_requires_input


def test_priming_from_a_real_value_works():
    node = serial(p=Primed()).with_input(X)
    assert jnp.allclose(node.init(input=X).p, X)


def test_serial_switches_from_value_to_shape_after_the_last_primer():
    def resize_apply(state, input):
        return state, jnp.sum(input)

    resize = Leaf(
        resize_apply, init=lambda input: input, name='resize').node

    def shape_init(node):
        return jnp.zeros_like(node.input)

    shaped = Leaf(
        lambda state, input: (state, input),
        init=shape_init,
        name='shaped',
    ).node

    model = serial(resize=resize, shaped=shaped).parameterize()
    state = model.init(input=X)
    assert state.shaped.shape == ()


# every transform that walks, one case each
CASES = {
    'serial': lambda: serial(p=Primed()).with_input(X),
    'composite': lambda: Composite(p=Primed())(lambda self, input: self.p(input),
                                          name='c'
                                          ).with_input(X),
    'batch': lambda: batch(Primed()).with_input(jnp.zeros((2, 3))),
    'stack': lambda: stack(Primed(), n=2).with_input(X),
    'ensemble': lambda: ensemble(Primed(), n=2).with_input(X),
    'repeat': lambda: repeat(Primed(), n=2).with_input(X),
    'at': lambda: at(Primed(), 'f').with_input(Struct(f=X)),
    'sum_junction': lambda: sum_junction(
        a=Primed()).with_input(X),
    'parallel': lambda: parallel(a=Primed()).with_input(bundle=Struct(a=X)),
    'residual': lambda: residual(Primed()).with_input(X),
    'remat': lambda: remat(Primed()).with_input(X),
    'state_reinit': lambda: state_reinit(Primed()).with_input(X),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_shape_alone_refuses_to_prime(name):
    """init() with no value supplied: the node is resolved, so a shape is
    available, and that is exactly when the zeros are tempting."""
    with pytest.raises(TypeError, match='real input value'):
        CASES[name]().init()
