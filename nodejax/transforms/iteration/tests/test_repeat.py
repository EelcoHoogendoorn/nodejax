"""Verifies weight-tied iteration transforms (`repeat`, `repeated`, and `iterated`)."""

import jax
import jax.numpy as jnp

from nodejax import Node, Leaf, repeat, stack
from nodejax.control import Gain, Integrator


def test_repeat():
    """Verifies that `repeat` applies one unstacked parameter tree `n` times sequentially."""
    r = repeat(Gain(), n=3).parameterize(scale=jnp.array(2.0))
    assert jnp.allclose(r.apply(1.0), 8.0)
    assert r.param.scale.shape == ()

    bound = Gain().parameterize(scale=jnp.array(3.0))
    assert jnp.allclose(repeat(bound, n=2).apply(1.0), 9.0)


def test_repeat_cyclic():
    """Verifies that `repeat` threads one unstacked state slot across all `n` iterations."""
    r = repeat(Integrator(), n=2).parameterize()
    state = r.init()
    assert jnp.shape(state) == ()
    state, out = r.apply(state, 1.0)
    assert jnp.allclose(state, 2.0) and jnp.allclose(out, 2.0)
    state, out = r.apply(state, 1.0)
    assert jnp.allclose(state, 6.0) and jnp.allclose(out, 6.0)


def test_repeat_allocates_one_apply_stream_per_step():
    def apply(input, rng):
        return input + jax.random.normal(rng.next())

    definition = repeat(Leaf(apply), n=4)
    assert definition.contract.apply_takes_rng

    key = jax.random.PRNGKey(0)
    first = definition.apply(0.0, rng=key)
    replay = definition.apply(0.0, rng=key)
    other = definition.apply(0.0, rng=jax.random.PRNGKey(1))
    assert jnp.array_equal(first, replay)
    assert not jnp.array_equal(first, other)


def Register() -> Node:
    """Primes state from the initial input and increments subsequent inputs by one."""
    def init(input):
        return input
    def apply(state, input):
        return state, input + 1.0
    return Leaf(apply, init=init, name='register')


def test_stack_init_walks_the_positions():
    """Verifies that `stack` primes each layer from the preceding layer's output while `repeat` primes once."""
    state = stack(Register().node, n=3).parameterize().init(input=jnp.array(1.0))
    assert jnp.allclose(state, jnp.array([1.0, 2.0, 3.0]))

    state = repeat(Register(), n=3).parameterize().init(input=jnp.array(1.0))
    assert jnp.shape(state) == () and jnp.allclose(state, 1.0)


def test_iterated_applies_one_node_to_the_same_input_n_times():
    """Verifies that `iterated` threads state across `n` steps while holding the step input fixed."""
    from nodejax import iterated

    def Accumulator() -> Node:
        def apply(state, input):
            total = state + input
            return total, total

        return Leaf(apply, init=lambda: jnp.zeros(()), name='accumulator').node

    threefold = iterated(Accumulator(), n=3).parameterize().initialize()
    threefold, output = threefold.apply(2.0)

    assert output == 6.0
    assert threefold.state == 6.0


def test_repeated_keeps_every_point() -> None:
    from nodejax import Leaf, repeated

    doubling = Leaf(lambda input: 2.0 * input, name='doubling')
    trajectory = repeated(doubling, n=3).parameterize().apply(jnp.asarray(1.0))
    assert jnp.allclose(trajectory, jnp.asarray([2.0, 4.0, 8.0]))


def test_repeated_runs_from_fresh_state_and_drops_it() -> None:
    """Verifies that `repeated` initializes fresh state per call, stacks outputs, and discards final state."""
    from nodejax import Node, repeated
    from nodejax.control import Integrator

    run = repeated(Integrator(), n=3)
    assert type(run) is Node and not run.cyclic
    trajectory = run.parameterize().apply(jnp.asarray(1.0))
    assert jnp.allclose(trajectory, jnp.asarray([1.0, 2.0, 4.0]))
    assert jnp.allclose(run.parameterize().apply(jnp.asarray(1.0)), trajectory)


def test_repeat_and_repeated_run_a_state_bound_step_from_its_state() -> None:
    from nodejax import repeated, repeat
    from nodejax.control import Integrator

    started = Integrator().parameterize().initialize().bind(state=jnp.asarray(10.0))
    trajectory = repeated(started, n=3).apply(jnp.asarray(1.0))
    assert jnp.allclose(trajectory, jnp.asarray([11.0, 22.0, 44.0]))
    replayed = repeated(started, n=3).specialize().parameterize()
    assert jnp.allclose(
        replayed.apply(jnp.asarray(1.0)),
        jnp.asarray([11.0, 22.0, 44.0]),
    )
    run, last = repeat(started, n=3).apply(jnp.asarray(1.0))
    assert last == 44.0 and run.state == 44.0
