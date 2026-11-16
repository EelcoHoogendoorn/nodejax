"""Verifies that walking transforms distinguish `node.input` shape specs from real sample values.

`node.input` materializes zeros of the resolved input spec so a walk can
propagate shapes through a composition, resolving each member's definition
against the preceding member's output. Those zero arrays represent shape
evidence rather than numerical data.

A node whose state starts at its first real sample declares
`init_requires_input` and refuses to initialize when only a bound shape is
available, preventing silent warm-starts from synthetic zeros across every
walking transform.
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
    """Builds a stateful leaf whose initial state equals its first sample."""
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
    """Verifies `init()` without a real sample raises `TypeError` even when shape-resolved."""
    with pytest.raises(TypeError, match='real input value'):
        CASES[name]().init()
