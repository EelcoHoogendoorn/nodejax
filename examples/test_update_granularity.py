"""Controlling running-statistic refresh granularity across steps, episodes, and frozen evaluation.

A Node that normalizes against its own running mean can refresh the estimate
that `apply` reads at three distinct rates:

    every step      single cyclic state slot (`Tracker`), reading prior steps immediately
    every episode   separate `live` and `snapshot` slots (`episodic`), copying `live` into
                    `snapshot` at `'episode'` boundaries while `live` continues accumulating
    never           `freeze(node, state)`, binding state to a fixed constant and removing cycling

`Tracker` and `freeze` share a single-slot definition whose enclosing transform
selects step-rate updates or frozen evaluation. `episodic` declares its
two-slot boundary transition locally via `boundary={'episode': hold}`, so
`scan(..., boundary='episode')` triggers the snapshot refresh by tag name
without referencing internal state field names.
"""

import jax.numpy as jnp

from nodejax import Leaf, Node, PNode, freeze, node, scan, scanned
from nodejax.struct import Struct

MOMENTUM = 0.5
SEQUENCE = jnp.ones(4)


@node
def Tracker(momentum: float = MOMENTUM) -> PNode:
    """Tracks an exponential moving average in one cyclic state slot and subtracts the prior estimate."""
    def init(node):
        return jnp.zeros_like(node.input)

    def apply(state, input):
        out = input - state
        return (1 - momentum) * state + momentum * input, out

    return Leaf(apply, init=init)


@node
def episodic(momentum: float = MOMENTUM) -> PNode:
    """Maintains separate `live` and `snapshot` moving averages, refreshing `snapshot` at `'episode'` boundaries.

    Each step subtracts `state.snapshot` while updating `state.live`. At an
    `'episode'` boundary, `hold` copies the carried `live` estimate into
    `snapshot` while leaving `live` carried across episodes by default.
    """
    def init(node):
        zero = jnp.zeros_like(node.input)
        return Struct(live=zero, snapshot=zero)

    def apply(state, input):
        out = input - state.snapshot
        live = (1 - momentum) * state.live + momentum * input
        return state.replace(live=live), out

    def hold(carried, init, decided):
        return decided.replace(snapshot=carried.live)

    return Leaf(apply, init=init,
                    boundary={'episode': hold})


def test_every_step_reads_its_own_episode() -> None:
    """Verifies that `Tracker` updates its subtracted mean on every step within an episode."""
    episode = scanned(Tracker()).with_input(SEQUENCE)
    out = episode.apply(SEQUENCE)

    # Outputs follow `1 - 0`, `1 - 0.5`, `1 - 0.75`, and `1 - 0.875`.
    assert jnp.allclose(out, jnp.array([1.0, 0.5, 0.25, 0.125]))
    assert not jnp.allclose(out, out[0])


def test_every_episode_is_fixed_within_and_moves_between() -> None:
    """Verifies that `episodic` holds `snapshot` constant within each episode and updates it from `live` between episodes."""
    run = scan(episodic(), boundary='episode').with_input(
        SEQUENCE).initialize()

    run, first = run(SEQUENCE)
    assert jnp.allclose(first, first[0])
    assert jnp.allclose(first, 1.0)

    run, second = run(SEQUENCE)
    assert jnp.allclose(second, second[0])
    # After four steps at momentum 0.5, `live` reaches 0.9375 and copies into `snapshot`.
    assert jnp.allclose(second, 1.0 - 0.9375)
    assert not jnp.allclose(first[0], second[0])


def test_never_is_freeze() -> None:
    """Verifies that `freeze` binds state to a constant, removes `cyclic`, and produces identical outputs on repeated calls."""
    step = episodic().with_input(jnp.asarray(1.0))
    held = step.init().replace(snapshot=jnp.asarray(0.25))

    pinned = freeze(step, held)
    assert not pinned.cyclic
    assert jnp.allclose(pinned.apply(1.0), 0.75)
    assert jnp.allclose(pinned.apply(1.0), pinned.apply(1.0))


def test_the_three_rates_disagree_as_they_should() -> None:
    """Verifies that per-step `Tracker` decays within an episode while `episodic` stays constant at its refreshed level."""
    fast = scanned(Tracker()).with_input(SEQUENCE)
    slow = scan(episodic(), boundary='episode').with_input(SEQUENCE)

    per_step = fast.apply(SEQUENCE)
    run, _ = slow.initialize()(SEQUENCE)
    _, per_episode = run(SEQUENCE)

    assert not jnp.allclose(per_step, per_step[0])
    assert jnp.allclose(per_episode, per_episode[0])
    assert not jnp.allclose(per_step[-1], per_episode[-1])
