"""repeat: weight-tied depth — one param, applied n times in sequence."""

import jax
import jax.numpy as jnp

from nodejax import Node, Leaf, repeat, stack
from nodejax.control import Gain, Integrator


def test_repeat():
    """One param applied n times — vs stack's per-layer params — and
    rebinding a bound PNode (param meaning unchanged)."""
    r = repeat(Gain(), n=3).parameterize(scale=jnp.array(2.0))
    assert jnp.allclose(r.apply(1.0), 8.0)
    assert r.param.scale.shape == ()          # no layer axis, unlike stack

    bound = Gain().parameterize(scale=jnp.array(3.0))
    assert jnp.allclose(repeat(bound, n=2).apply(1.0), 9.0)


def test_repeat_cyclic():
    """One state slot, threaded through the iterations exactly as through
    calls: repeat is apply in a loop, and the state keeps the inner's own
    shape."""
    r = repeat(Integrator(), n=2).parameterize()
    state = r.init()
    assert jnp.shape(state) == ()          # no position axis: same node, looped
    state, out = r.apply(state, 1.0)
    # two iterations: 0+1 -> 1, then 1 plus the first iteration's output -> 2
    assert jnp.allclose(state, 2.0) and jnp.allclose(out, 2.0)
    state, out = r.apply(state, 1.0)
    # the state carries into the next call: 2+1 -> 3, then 3+3 -> 6
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
    """Primes its state from the first value it meets; apply increments the
    signal and holds the primed value still."""
    def init(input):
        return input
    def apply(state, input):
        return state, input + 1.0
    return Leaf(apply, init=init, name='register')


def test_stack_init_walks_the_positions():
    """stack's layers run in SEQUENCE, so a priming init at layer k boots
    from the signal as it arrives there, not from the raw input n times: the
    walk serial's init does, done with one node. ensemble legitimately primes
    every member from the same input, which is what makes it the parallel
    one, and repeat needs no walk at all: one state, primed once."""
    state = stack(Register().node, n=3).parameterize().init(input=jnp.array(1.0))
    assert jnp.allclose(state, jnp.array([1.0, 2.0, 3.0]))

    state = repeat(Register(), n=3).parameterize().init(input=jnp.array(1.0))
    assert jnp.shape(state) == () and jnp.allclose(state, 1.0)


def test_iterated_applies_one_node_to_the_same_input_n_times():
    """Function iteration on the state: the input stays constant, unlike
    repeat's output chaining. The last output returns; aux stacks."""
    from nodejax import iterated

    def Accumulator() -> Node:
        def apply(state, input):
            total = state + input
            return total, total

        return Leaf(apply, init=lambda: jnp.zeros(()), name='accumulator').node

    threefold = iterated(Accumulator(), n=3).parameterize().initialize()
    threefold, output = threefold.apply(2.0)

    assert output == 6.0                       # 0 + 2 + 2 + 2, last output
    assert threefold.state == 6.0              # state threaded through


def test_repeated_keeps_every_point() -> None:
    from nodejax import Leaf, repeated

    doubling = Leaf(lambda input: 2.0 * input, name='doubling')
    trajectory = repeated(doubling, n=3).parameterize().apply(jnp.asarray(1.0))
    assert jnp.allclose(trajectory, jnp.asarray([2.0, 4.0, 8.0]))


def test_repeated_runs_from_fresh_state_and_drops_it() -> None:
    """The past-complete form of repeat: a cyclic step starts from fresh
    state on every call and the result has no state slot."""
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
