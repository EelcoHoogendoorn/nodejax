"""Ambient construction arguments: dynamic scope for the def-building
stage — declared at the definition site, supplied at one point of use,
construction-time only."""

import pytest

from nodejax import ambient, node_def


@ambient
def block(dt, gain=1.0):
    return node_def(lambda input: gain * input * dt, name='block')


def test_scope_fills_unbound():
    with ambient(dt=0.5):
        nd = block()
    assert nd.apply(2.0) == 1.0


def test_explicit_always_wins():
    with ambient(dt=0.5, gain=3.0):
        nd = block(0.1)                        # positional dt beats scope
    assert nd.apply(1.0) == pytest.approx(0.3)


def test_scopes_nest_inner_wins():
    with ambient(dt=1.0):
        with ambient(dt=0.25):
            inner = block()
        outer = block()
    assert inner.apply(1.0) == 0.25
    assert outer.apply(1.0) == 1.0


def test_outside_scope_nothing_becomes_optional():
    with pytest.raises(TypeError):
        block()                                # dt required, no scope


def test_scope_restores_on_exit():
    with ambient(dt=1.0):
        pass
    with pytest.raises(TypeError):
        block()
