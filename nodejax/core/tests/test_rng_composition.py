"""RNG keys reaching a node under compositions of the basic transforms.

Four ways a node can need keys, crossed with the ways scan, scanned and
batch nest. The individual cases were all covered somewhere; the CROSSINGS
were not, and that is where the gaps were: three of the thirty-two failed,
each in a different place, and each only when a node's INIT draws while its
apply does not.

That column is the whole story. When a node draws at apply as well, a
transform above it sees a non-empty apply plan and routes a key by accident, which
is why the bugs hid: every case that needed a key for its init also happened
to need one for its apply, except the ones nobody had written.

Two rules close it, and they differ in whether the need changes on the way up.

  SAME NEED, ONE LEVEL UP. If anything beneath draws at apply, a key has to
  travel through this node to reach it, so this node wants one at apply too.
  Unchanged at every level, which is why it can be read off the tree instead
  of restated by each transform. Three transforms restated it and one dropped
  it, and a property read off `members` cannot be dropped without dropping the
  member.

  A DIFFERENT NEED, ONE LEVEL UP. scanned builds a start state on every call,
  and a claimed boundary builds a fresh init on every call, so a need the
  inner declares for its INIT becomes a need this node has at APPLY. The
  channel changes, so no reading of the tree can find it: nothing beneath has
  an apply that draws.
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
    """The column that broke. An RNG key is wanted once, to build state, and
    never again, so nothing about this node's apply says a word about rng."""
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
    """Thirty-two crossings, and every one has to run. The assertion is that
    it runs at all: a key that fails to arrive raises rather than returning
    something silently wrong, so reaching the end IS the property."""
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
    """A transform cannot lose the need by forgetting to restate it, because
    the property is derived from `members` and a wrapper's inner is one.

    scan used to restate it and drop it, so scan(draws)'s apply plan was
    False and the level above routed nothing."""
    leaf = apply_draws()
    assert bool(leaf.contract.apply_takes_rng)
    for wrap in (scan, scanned, lambda d: batch(d, n=2), state_reinit):
        assert bool(wrap(leaf).contract.apply_takes_rng), wrap
    assert bool(scanned(scan(batch(leaf, n=2))).contract.apply_takes_rng)  # through a tower


def test_an_init_need_becomes_an_apply_need():
    """The half no reading of the tree can find. This node's apply draws nothing;
    the need is on its state slot, and scanned turns that into an apply
    need because it builds a start state on every call."""
    leaf = init_draws()
    assert not bool(leaf.contract.apply_takes_rng)                      # its apply wants nothing
    assert bool(leaf.contract.init_takes_rng)                            # its init does

    assert bool(scanned(leaf).contract.apply_takes_rng)                 # converted
    assert bool(scan(state_reinit(leaf, 'ep'), boundary='ep').contract.apply_takes_rng)
    assert not bool(scan(leaf).contract.apply_takes_rng)                # no init runs at apply here


def test_each_batch_element_draws_its_own():
    """What the conversion buys, and the case that failed loudest: batch read
    the wrong role metadata and failed to split the explicit stream over its
    declared axis."""
    node = batch(scanned(init_draws()), n=2).parameterize()
    out = node.apply(rng=KEY, x=jnp.zeros((2, 4)))

    assert out.shape == (2, 4)
    assert not jnp.allclose(out[0], out[1])              # two elements, two draws


def test_a_deterministic_sibling_does_not_move_the_stream():
    """Why the need is declared at all rather than a key threaded everywhere.
    Only nodes that consume get splits, so inserting one that does not leaves
    its siblings bit-identical. Unconditional threading cannot promise this,
    and it is the same claim the library makes about composition generally."""
    from nodejax import serial

    def Quiet():
        # passes the whole bundle on, since the drawing member downstream is
        # typed by FIELDS and a bare array could not satisfy it
        def apply(state, input):
            return state, input

        return Leaf(apply, init=lambda: jnp.zeros(()), name='quiet').node

    alone = scanned(serial(a=apply_draws())).parameterize().apply(
        jnp.zeros(4), rng=KEY)
    beside = scanned(serial(q=Quiet(), a=apply_draws())).parameterize().apply(
        jnp.zeros(4), rng=KEY)

    assert jnp.allclose(alone, beside)


def test_the_rng_frame_does_not_project_to_a_step():
    """scan's init takes the sequence's first data element to build step state.

    The RNG frame is a separate pytree with its own role plan, not a data field
    on that time axis. Treating it as sequence data once produced a scalar key
    fragment at one nesting level and indexed that scalar at the next.

    The three-level tower in the chunk comparison is what found it, which is
    the case a matrix of two levels could not reach."""
    node = scanned(scan(scan(state_reinit(init_draws(), 'ep'), boundary='ep')))
    bound = node.parameterize()

    out = bound.apply(rng=KEY, x=jnp.zeros((2, 3, 4)))
    assert out.shape == (2, 3, 4)

    # and the draw is real: another key, another run
    other = bound.apply(rng=jax.random.PRNGKey(9), x=jnp.zeros((2, 3, 4)))
    assert not jnp.allclose(out, other)
