"""Verifies PRNG routing across nested `scan`, `scanned`, `state_reinit`, and `batch` compositions.

Tests four leaf PRNG profiles (deterministic, apply-drawing, init-drawing,
and both-drawing) crossed with eight transform towers:

1. Direct propagation: when any inner member draws at `apply`, enclosing
   wrappers inherit `apply_takes_rng` from `members` and forward the stream.
2. Role conversion: `scanned` and `scan(..., boundary=...)` run `init` inside
   `apply`, converting an inner `init_takes_rng` requirement into an outer
   `apply_takes_rng` requirement.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, Leaf, scan, scanned, state_reinit, batch
from nodejax.struct import Struct

KEY = jax.random.PRNGKey(0)


def no_rng() -> Node:
    def apply(state, x):
        return state + x, state

    return Leaf(apply, init=lambda: jnp.zeros(()), name='none').node


def apply_draws() -> Node:
    def apply(state, x, rng):
        return state, jax.random.normal(rng.next(), ())

    return Leaf(apply, init=lambda: jnp.zeros(()), name='apply_rng').node


def init_draws() -> Node:
    """Builds a cyclic leaf that draws a PRNG key during `init` only."""
    def init(rng):
        return jax.random.normal(rng.next(), ())

    def apply(state, x):
        return state + x, state

    return Leaf(apply, init=init, name='init_rng').node


def both_draw() -> Node:
    def init(rng):
        return jax.random.normal(rng.next(), ())

    def apply(state, x, rng):
        return state, state + jax.random.normal(rng.next(), ())

    return Leaf(apply, init=init, name='both').node


NODES = [no_rng, apply_draws, init_draws, both_draw]
TOWERS = [
    ('scanned', lambda f: scanned(f()), (4,), False),
    ('scan', lambda f: scan(f()), (4,), True),
    ('scan claiming a boundary',
     lambda f: scan(state_reinit(f(), 'ep'), boundary='ep'), (4,), True),
    ('scanned over scan', lambda f: scanned(scan(f())), (3, 4), False),
    ('scanned over a claimed scan',
     lambda f: scanned(scan(state_reinit(f(), 'ep'), boundary='ep')), (3, 4), False),
    ('batch over scanned', lambda f: batch(scanned(f()), n=2), (2, 4), False),
    ('scanned over batch', lambda f: scanned(batch(f(), n=2)), (4, 2), False),
    ('batch over scan', lambda f: batch(scan(f()), n=2), (2, 4), True),
]


@pytest.mark.parametrize('tower,build,shape,cyclic', TOWERS,
                         ids=[t[0] for t in TOWERS])
@pytest.mark.parametrize('factory', NODES, ids=[f.__name__ for f in NODES])
def test_rng_reaches_the_node(tower, build, shape, cyclic, factory):
    """Verifies all thirty-two `(tower, leaf)` PRNG crossings execute without missing keys."""
    node = build(factory)
    bound = node.parameterize()
    key = {'rng': KEY} if bool(node.contract.apply_takes_rng) else {}

    if cyclic:
        state_in = {'rng': KEY} if bool(node.contract.init_takes_rng) else {}
        _, out = bound.initialize(**state_in)(x=jnp.zeros(shape), **key)
    else:
        out = bound.apply(x=jnp.zeros(shape), **key)
    assert out is not None


def test_an_apply_need_travels_up_unchanged():
    """Verifies `apply_takes_rng` propagates upward from `members` through nested wrappers."""
    leaf = apply_draws()
    assert bool(leaf.contract.apply_takes_rng)
    for wrap in (scan, scanned, lambda d: batch(d, n=2), state_reinit):
        assert bool(wrap(leaf).contract.apply_takes_rng), wrap
    assert bool(scanned(scan(batch(leaf, n=2))).contract.apply_takes_rng)  # through a tower


def test_an_init_need_becomes_an_apply_need():
    """Verifies `scanned` and boundary-claiming `scan` convert `init_takes_rng` into `apply_takes_rng`."""
    leaf = init_draws()
    assert not bool(leaf.contract.apply_takes_rng)                      # its apply wants nothing
    assert bool(leaf.contract.init_takes_rng)                            # its init does

    assert bool(scanned(leaf).contract.apply_takes_rng)                 # converted
    assert bool(scan(state_reinit(leaf, 'ep'), boundary='ep').contract.apply_takes_rng)
    assert not bool(scan(leaf).contract.apply_takes_rng)                # no init runs at apply here


def test_each_batch_element_draws_its_own():
    """Verifies `batch(scanned(init_draws()))` splits the `apply` stream independently per element."""
    node = batch(scanned(init_draws()), n=2).parameterize()
    out = node.apply(rng=KEY, x=jnp.zeros((2, 4)))

    assert out.shape == (2, 4)
    assert not jnp.allclose(out[0], out[1])              # two elements, two draws


def test_a_deterministic_sibling_does_not_move_the_stream():
    """Verifies inserting a deterministic sibling leaves stochastic draws bit-identical."""
    from nodejax import serial

    def Quiet():
        # forwards the bundle unchanged so downstream field-typed members match
        def apply(state, input):
            return state, input

        return Leaf(apply, init=lambda: jnp.zeros(()), name='quiet').node

    alone = scanned(serial(a=apply_draws())).parameterize().apply(
        jnp.zeros(4), rng=KEY)
    beside = scanned(serial(q=Quiet(), a=apply_draws())).parameterize().apply(
        jnp.zeros(4), rng=KEY)

    assert jnp.allclose(alone, beside)


def test_the_rng_frame_does_not_project_to_a_step():
    """Verifies three-level `scanned` / `scan` nesting routes PRNG streams without time-axis indexing."""
    node = scanned(scan(scan(state_reinit(init_draws(), 'ep'), boundary='ep')))
    bound = node.parameterize()

    out = bound.apply(rng=KEY, x=jnp.zeros((2, 3, 4)))
    assert out.shape == (2, 3, 4)

    # and the draw is real: another key, another run
    other = bound.apply(rng=jax.random.PRNGKey(9), x=jnp.zeros((2, 3, 4)))
    assert not jnp.allclose(out, other)
