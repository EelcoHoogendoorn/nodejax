"""Verifies `batch` mapping over a leading input axis with shared parameters and per-element state."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import batch, Leaf, materialize, nn, stack
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_batch():
    b = batch(Gain()).parameterize(scale=jnp.array(2.0))
    out = b.apply(jnp.array([1.0, 2.0, 4.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 8.0]))


def test_batch_of_batch_accepts_a_nested_struct_input():
    input_spec = Struct(
        state=Struct(
            angle=jnp.zeros(()),
            velocity=jnp.zeros(()),
        ),
        memory=jnp.zeros(4),
    )

    def parameterize():
        return jnp.asarray(2.0)

    def apply(param, input):
        return param * (
            input.state.angle
            + input.state.velocity
            + jnp.sum(input.memory)
        )

    critic = Leaf(
        apply,
        param=parameterize,
        apply_input_spec=input_spec,
        name='critic',
    )
    model = batch(batch(critic, n=3), n=2).parameterize()
    angle = jnp.arange(6.0).reshape(2, 3)
    input = Struct(
        state=Struct(
            angle=angle,
            velocity=jnp.ones((2, 3)),
        ),
        memory=jnp.ones((2, 3, 4)),
    )
    output = jax.jit(model.apply)(input)
    declared = materialize(model.contract.input_spec).input

    assert output.shape == (2, 3)
    assert jnp.allclose(output, 2.0 * (angle + 5.0))
    assert declared.state.angle.shape == (2, 3)
    assert declared.state.velocity.shape == (2, 3)
    assert declared.memory.shape == (2, 3, 4)


def test_batch_cyclic():
    b = batch(Integrator(), n=3).parameterize()
    state = b.init()
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    state, out = b.apply(state, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0]))


def test_batch_primes_each_state_from_its_own_row():
    """Verifies that `batch` maps priming initialization across batch elements rather than tiling row zero."""
    def init(input):
        return input

    def apply(state, input):
        return state, input

    model = batch(Leaf(apply, init=init), n=3)
    values = jnp.array([1.0, 2.0, 4.0])
    state = model.init(input=values)
    assert jnp.allclose(state, values)


def test_batch_binds_its_named_axis_during_construction() -> None:
    """Verifies that `batch` binds its collective axis name during construction-time input propagation."""
    def init(input):
        return input

    def apply(state, input):
        return input, input

    primed = Leaf(apply, init=init, name='primed')
    layer = primed >> nn.BatchNorm(momentum=0.1)
    values = jnp.arange(6.0).reshape(2, 3)
    pipe = stack(layer, n=2) >> nn.Linear(2)
    model = batch(pipe).with_input(values).parameterize(
        rng=jax.random.PRNGKey(0)
    )

    initialized = model.initialize(input=values)
    stacked_state = initialized.state.stack_primed_batch_norm

    assert initialized.param.linear.w.shape == (3, 2)
    assert stacked_state.primed.shape == (2, 2, 3)
    assert stacked_state.batch_norm.mean.shape == (2, 3)
    assert stacked_state.batch_norm.var.shape == (2, 3)


def test_batch_nonpriming_input_binds_shape_without_forwarding_value():
    """Verifies that a non-priming `init` resolves element shape from a supplied batch without reading values."""
    def init(node):
        return jnp.zeros_like(node.input)

    def apply(state, input):
        return state, input

    model = batch(Leaf(apply, init=init))
    state = model.init(input=jnp.ones((3, 2)))
    assert state.shape == (3, 2)
    assert jnp.allclose(state, 0.0)


def test_batch_single_batch_state():

    def pop_apply(param, state, input):
        total = jax.lax.psum(input, 'batch')
        new_state = Struct(count=state.count + total)
        return new_state, input * param.scale

    def pop_init(param):
        return Struct(count=jnp.array(0.0))

    pop_node = Leaf(pop_apply, init=pop_init, param=lambda: Struct(scale=2.0), tags={'single_batch_state'})

    b = batch(pop_node).with_input(jnp.array([1.0, 2.0, 3.0]))
    m = b.parameterize()
    state = m.init()
    assert state.count.shape == ()
    state, out = m.apply(state, jnp.array([1.0, 2.0, 3.0]))
    assert state.count == 6.0
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0]))


def test_batch_splits_one_boundary_key_per_element():
    """Verifies that `batch` preserves `apply_takes_rng` and splits one boundary PRNG key across elements."""
    def apply(x: jax.Array, rng) -> jax.Array:
        return x + jax.random.normal(rng.next(), jnp.shape(x))
    noisy = Leaf(apply)
    assert noisy.contract.apply_takes_rng

    b = batch(noisy, n=4).with_input(
        bundle=Struct(x=jnp.zeros((4, 3))))
    assert b.contract.apply_takes_rng
    out = b.apply(x=jnp.ones((4, 3)), rng=jax.random.PRNGKey(1))
    assert out.shape == (4, 3)
    assert not jnp.allclose(out[0], out[1])


def test_a_random_init_draws_once_per_batch_element():
    """Verifies that stochastic state initialization draws an independent subkey for each batch element."""
    noisy = Leaf(
        lambda state, input: (state, state + input),
        init=lambda rng: jax.random.normal(rng.next(), (2,)),
        name='noisy',
    )
    state = batch(noisy, n=3).parameterize().initialize(rng=jax.random.PRNGKey(0)).state

    assert state.shape == (3, 2)
    assert not jnp.allclose(state[0], state[1])
    assert not jnp.allclose(state[1], state[2])


def test_state_inputs_carry_the_batch_axis_at_init():
    """Verifies that `batch` maps `init` across batched state-initialization fields."""
    counter = Leaf(
        lambda state, input: (state + input, state),
        init=lambda initial: initial,
        name='counter',
    )
    starts = jnp.asarray([1.0, 2.0, 3.0])
    node = batch(counter, n=3).parameterize().initialize(initial=starts)

    assert jnp.allclose(node.state, starts)
