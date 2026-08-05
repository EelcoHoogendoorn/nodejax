"""scan: internalize the state loop — a stepper becomes a sequence function.

With persist=, the state splits into SLOW slots (carried across
episodes) and FAST slots (re-initialized at every episode start): the
apply becomes an episode, and the mapping decides what outlives it.
"""

import jax.numpy as jnp

from nodejax import NodeDef, node_def, scan
from nodejax.struct import Struct
from nodejax.examples import integrator_def


def test_scan_transform():
    """PCN -> PN: sequence-level node with internalized state."""
    seq = scan(integrator_def())
    assert isinstance(seq, NodeDef) and not seq.cyclic
    node = seq.parameterize(gain=jnp.array(1.0))
    outs = node.apply(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_scan_persist_slow_fast():
    """Named slots carry across episodes; unmatched slots re-initialize
    fresh at every episode start."""
    def acc_def():
        def init():
            return Struct(count=jnp.zeros(()), register=jnp.zeros(()))

        def apply(state, input):
            new = Struct(count=state.count + 1.0, register=state.register + input)
            return new, new.count

        return node_def(apply, init=init, name='acc')

    seq = scan(acc_def(), persist=('count',))
    assert seq.cyclic                             # slow state stays outside

    s1, counts = seq.apply(seq.init(), jnp.ones(5))
    assert s1.count == 5.0 and s1.register == 5.0

    s2, counts = seq.apply(s1, jnp.ones(5))
    assert s2.count == 10.0                       # slow: carried, kept counting
    assert s2.register == 5.0                     # fast: fresh, re-accumulated


def test_scan_persist_frozen_refresh():
    """{'frozen': 'stats'} refreshes a read-copy from the carried
    accumulator at episode start — within an episode, reads never see
    that episode's own updates (the target-network shape)."""
    def two_slot_def():
        def init():
            return Struct(frozen=jnp.zeros(()), stats=jnp.zeros(()))

        def apply(state, input):
            new = Struct(frozen=state.frozen, stats=state.stats + input)
            return new, state.frozen

        return node_def(apply, init=init, name='twoslot')

    seq = scan(two_slot_def(), persist={'frozen': 'stats', 'stats': 'stats'})

    s1, reads = seq.apply(seq.init(), jnp.ones(4))
    assert jnp.all(reads == 0.0)                  # episode 1 reads the empty copy
    assert s1.stats == 4.0

    s2, reads = seq.apply(s1, jnp.ones(4))
    assert jnp.all(reads == 4.0)                  # frozen := carried stats, held
    assert s2.stats == 8.0                        # accumulator keeps growing
