"""Verifies `ambient` dynamic construction scopes, override precedence, and scope exit."""

import pytest

from nodejax import node, ambient, Leaf


@node
def block(dt: float, gain: float=1.0):
    return Leaf(lambda input: gain * input * dt)


def test_scope_fills_unbound():
    with ambient(dt=0.5):
        node = block()
    assert node.apply(2.0) == 1.0


def test_explicit_always_wins():
    with ambient(dt=0.5, gain=3.0):
        node = block(0.1)                        # Positional argument overrides ambient scope.
    assert node.apply(1.0) == pytest.approx(0.3)


def test_scopes_nest_inner_wins():
    with ambient(dt=1.0):
        with ambient(dt=0.25):
            inner = block()
        outer = block()
    assert inner.apply(1.0) == 0.25
    assert outer.apply(1.0) == 1.0


def test_outside_scope_nothing_becomes_optional():
    """Verifies `block()` remains generic when `dt` has no ambient or explicit value."""
    assert block().generic


def test_scope_restores_on_exit():
    with ambient(dt=1.0):
        pass
    assert block().generic                       # Ambient binding expires when the block exits.
