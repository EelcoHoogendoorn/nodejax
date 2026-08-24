"""How often a node's running statistic is allowed to change.

A node that reads its own statistics has to answer that question, and
there are three useful answers. The same running mean is written twice
below, once with one state slot and once with the read and write slots
separated, and each answer follows from what encloses it:

    every step      ordinary cyclic state: a step reads what earlier steps wrote
    every episode   read and write slots split, the snapshot refreshed by the
                    node's own boundary action while the live copy accumulates
    never           freeze(node, state): the slot is pinned

The first and third are the SAME node under a transform: Tracker is written
once, with one state slot and no idea that any of this is happening, and what
changes is what encloses it. The middle one is a second node, written out,
because the library ships no transform for it. The library used to, and it
paid for the convenience with a boundary action that read one slot of the
carry and wrote another, which is per-attribute metadata on a per-node
mechanism.

WHAT SURVIVES THE HAND-WRITING, and it is the part worth having: the boundary
logic stays inside the node, over slot names that never leave it. scan still
knows only WHERE the boundary is and says so by claiming a tag. No mapping is
spelled at the scan, and no caller names a slot.
"""

import jax.numpy as jnp

from nodejax import Node, node, Leaf, scan, scanned, freeze
from nodejax.struct import Struct

MOMENTUM = 0.5
SEQUENCE = jnp.ones(4)          # four steps, so the drift is easy to read


@node
def Tracker(momentum: float=MOMENTUM) -> Node:
    """Running mean, ONE slot: the value a step reads is the value every
    earlier step in the same episode has already moved."""
    def init(node):
        return jnp.zeros_like(node.input)

    def apply(state, input):
        out = input - state                                   # read
        return (1 - momentum) * state + momentum * input, out  # write

    return Leaf(apply, init=init)


@node
def episodic(momentum: float=MOMENTUM) -> Node:
    """The same running mean, read at the episode rate: the live copy goes on
    accumulating and reads come from the snapshot the boundary took.

    TWO SLOTS, written by hand. The read comes off `snapshot` and the write
    lands on `live`, so no step of an episode sees what earlier steps of the
    same episode contributed. `hold` is this node's own boundary action, over
    its own slot names, and it is the whole of the arrangement: at the tag it
    names, the snapshot moves up to where live stood.

    Carrying is the default, so `live` surviving the episode needs no word.
    Only the snapshot's move does."""
    def init(node):
        zero = jnp.zeros_like(node.input)
        return Struct(live=zero, snapshot=zero)    # cold, the two agree

    def apply(state, input):
        out = input - state.snapshot                            # read, held
        live = (1 - momentum) * state.live + momentum * input   # write, live
        return state.replace(live=live), out

    def hold(carried, init, decided):
        return decided.replace(snapshot=carried.live)

    return Leaf(apply, init=init,
                    boundary={'episode': hold})


def test_every_step_reads_its_own_episode():
    """One slot: the output decays within the episode, because each step
    subtracts a mean the earlier steps already pulled up."""
    episode = scanned(Tracker()).with_input(SEQUENCE)
    out = episode.apply(SEQUENCE)

    # 1 - 0, 1 - 0.5, 1 - 0.75, 1 - 0.875
    assert jnp.allclose(out, jnp.array([1.0, 0.5, 0.25, 0.125]))
    assert not jnp.allclose(out, out[0])            # the read moved underneath


def test_every_episode_is_fixed_within_and_moves_between():
    """The snapshot refreshes from the carried live copy at episode start and
    holds for the whole episode, so every step of one episode reads the same
    value and the next episode reads a new one."""
    run = scan(episodic(), boundary='episode').with_input(
        SEQUENCE).initialize()

    run, first = run(SEQUENCE)
    assert jnp.allclose(first, first[0])           # fixed across the episode
    assert jnp.allclose(first, 1.0)                # the snapshot started at zero

    run, second = run(SEQUENCE)
    assert jnp.allclose(second, second[0])         # fixed again, at a new value
    # stats reached 0.9375 over the first episode; the snapshot now holds it
    assert jnp.allclose(second, 1.0 - 0.9375)
    assert not jnp.allclose(first[0], second[0])   # the rate is per EPISODE


def test_never_is_freeze():
    """freeze pins the slot outright: the node stops being cyclic, and
    repeated applies are identical because nothing accumulates at all."""
    step = episodic().with_input(jnp.asarray(1.0))
    held = step.init().replace(snapshot=jnp.asarray(0.25))

    pinned = freeze(step, held)
    assert not pinned.cyclic                       # no state slot survives
    assert jnp.allclose(pinned.apply(1.0), 0.75)
    assert jnp.allclose(pinned.apply(1.0), pinned.apply(1.0))


def test_the_three_rates_disagree_as_they_should():
    """The point of the file, in one place: same node, same input, three
    refresh rates, three different second episodes."""
    fast = scanned(Tracker()).with_input(SEQUENCE)
    slow = scan(episodic(), boundary='episode').with_input(SEQUENCE)

    per_step = fast.apply(SEQUENCE)                       # decays within
    run, _ = slow.initialize()(SEQUENCE)
    _, per_episode = run(SEQUENCE)                        # flat, at a new level

    assert not jnp.allclose(per_step, per_step[0])        # moves within
    assert jnp.allclose(per_episode, per_episode[0])      # flat within
    assert not jnp.allclose(per_step[-1], per_episode[-1])
