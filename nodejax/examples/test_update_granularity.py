"""How often a node's running statistic is allowed to change.

A node that reads its own statistics has to answer that question, and
there are three useful answers. The same running mean is written twice
below, once with one state slot and once with the read and write slots
separated, and each answer follows from what encloses it:

    every step      one slot: a step reads what earlier steps wrote
    every episode   two slots, refreshed by scan's persist mapping
    never           freeze(node, state): the slot is pinned

The middle answer is the one with no spelling of its own. Freezing stops
the accumulation, and ordinary cyclic state lets step 2 read what step 1
just contributed. Keeping a value fixed across an episode while it still
accumulates underneath needs both slots.

Which is why the rate is not the node's to declare. The node says only
that reads and writes go to different slots. The enclosing scan says how
often one refreshes from the other.
"""

import jax.numpy as jnp

from nodejax import node_def, scan, freeze
from nodejax.struct import Struct

MOMENTUM = 0.5
SEQUENCE = jnp.ones(4)          # four steps, so the drift is easy to read


def Tracker(momentum=MOMENTUM):
    """Running mean, ONE slot: the value a step reads is the value every
    earlier step in the same episode has already moved."""
    def init(ndef):
        return jnp.zeros_like(ndef.input)

    def apply(state, input):
        out = input - state                                   # read
        return (1 - momentum) * state + momentum * input, out  # write

    return node_def(apply, init=init, name='tracker')


def SplitTracker(momentum=MOMENTUM):
    """The same running mean with the READ and WRITE slots split: reads take
    `frozen`, writes accumulate into `stats`. Nothing here refreshes frozen
    from stats — the node states the separation, never the rate."""
    def init(ndef):
        zero = jnp.zeros_like(ndef.input)
        return Struct(frozen=zero, stats=zero)

    def apply(state, input):
        out = input - state.frozen                            # read: fixed slot
        stats = (1 - momentum) * state.stats + momentum * input
        return Struct(frozen=state.frozen, stats=stats), out  # write: live slot

    return node_def(apply, init=init, name='split')


def test_every_step_reads_its_own_episode():
    """One slot: the output decays within the episode, because each step
    subtracts a mean the earlier steps already pulled up."""
    episode = scan(Tracker()).with_input(SEQUENCE)
    out = episode.apply(SEQUENCE)

    # 1 - 0, 1 - 0.5, 1 - 0.75, 1 - 0.875
    assert jnp.allclose(out, jnp.array([1.0, 0.5, 0.25, 0.125]))
    assert not jnp.allclose(out, out[0])            # the read moved underneath


def test_every_episode_is_fixed_within_and_moves_between():
    """Split slots plus persist: frozen refreshes from the carried stats at
    episode start and holds for the whole episode, so every step of one
    episode reads the same value, and the next episode reads a new one."""
    episode = scan(SplitTracker(),
                   persist={'frozen': 'stats', 'stats': 'stats'}).with_input(SEQUENCE)
    state = episode.init()

    state, first = episode.apply(state, SEQUENCE)
    assert jnp.allclose(first, first[0])           # fixed across the episode
    assert jnp.allclose(first, 1.0)                # frozen started at zero

    state, second = episode.apply(state, SEQUENCE)
    assert jnp.allclose(second, second[0])         # fixed again, at a new value
    # stats reached 0.9375 over the first episode; frozen now reads it
    assert jnp.allclose(second, 1.0 - 0.9375)
    assert not jnp.allclose(first[0], second[0])   # the rate is per EPISODE


def test_never_is_freeze():
    """freeze pins the slot outright: the node stops being cyclic, and
    repeated applies are identical because nothing accumulates at all."""
    step = SplitTracker().with_input(jnp.asarray(1.0))
    held = step.init().replace(frozen=jnp.asarray(0.25))

    pinned = freeze(step, held)
    assert not pinned.ndef.cyclic                  # no state slot survives
    assert jnp.allclose(pinned.apply(1.0), 0.75)
    assert jnp.allclose(pinned.apply(1.0), pinned.apply(1.0))


def test_the_three_rates_disagree_as_they_should():
    """The point of the file, in one place: same node, same input, three
    refresh rates, three different second episodes."""
    fast = scan(Tracker()).with_input(SEQUENCE)
    slow = scan(SplitTracker(),
                persist={'frozen': 'stats', 'stats': 'stats'}).with_input(SEQUENCE)

    per_step = fast.apply(SEQUENCE)                       # decays within
    state, _ = slow.apply(slow.init(), SEQUENCE)
    _, per_episode = slow.apply(state, SEQUENCE)          # flat, at a new level

    assert not jnp.allclose(per_step, per_step[0])        # moves within
    assert jnp.allclose(per_episode, per_episode[0])      # flat within
    assert not jnp.allclose(per_step[-1], per_episode[-1])
