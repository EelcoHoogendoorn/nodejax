"""Verifies `stack` scanning sequentially across `n` layers with independent per-layer parameters and state."""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import stack, repeat, scan, scanned, nn, Leaf, train_step
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def _warm_noise():
    def init(input):
        return input

    def apply(state, input, rng):
        output = input + jax.random.normal(rng.next(), input.shape)
        return state, output

    return Leaf(apply, init=init, name='warm_noise').node


def test_stack():
    s = stack(Gain(), n=2).bind(Struct(scale=jnp.array([2.0, 3.0])))
    assert jnp.allclose(s.apply(1.0), 6.0)


def test_a_plain_layer_stacks_without_inventing_storage():
    plain = Leaf(lambda input: input + 1, name='increment')
    layered = stack(plain, n=3).parameterize()

    assert layered.param == ()
    assert not layered.cyclic
    assert layered.apply(0) == 3


def test_per_layer_values_are_a_param_tree_not_a_construction():
    """Verifies that finished per-layer parameter arrays enter `stack` via `bind`."""
    s = stack(Gain(), n=3).bind(Struct(scale=jnp.array([2.0, 3.0, 4.0])))
    assert s.param.scale.shape == (3,)
    assert jnp.allclose(s.apply(1.0), 24.0)


def test_n_draws_a_layer_per_key():
    """Verifies that `stack` splits its `parameterize` PRNG key into `n` distinct layer draws."""
    s = stack(nn.Linear(4), n=3).with_input(jnp.zeros(4)).parameterize(
        rng=jax.random.PRNGKey(0))
    assert s.param.w.shape == (3, 4, 4)
    assert not jnp.allclose(s.param.w[0], s.param.w[1])


def test_n_tiles_a_value_the_caller_gives_once():
    """Verifies that `parameterize` tiles scalar constructor arguments into `n` independent layer slots."""
    s = stack(Gain(), n=3).parameterize(scale=2.0)
    assert s.param.scale.shape == (3,)
    assert jnp.allclose(s.param.scale, 2.0)
    assert jnp.allclose(s.apply(1.0), 8.0)


def test_n_tiles_the_defaults_too():
    """Verifies that defaulted parameter fields tile across all `n` layers."""
    s = scanned(stack(Integrator(), n=2)).parameterize()
    assert s.param.decay.shape == (2,)
    assert jnp.allclose(s.apply(jnp.ones(3)), jnp.array([1.0, 3.0, 6.0]))


def test_the_state_side_agrees_with_the_depth():
    """Verifies that stacked `param` and `state` trees share the same leading layer dimension `n`."""
    s = stack(Integrator(), n=4).parameterize()
    assert s.param.decay.shape == (4,)
    assert s.init().shape == (4,)


def test_primed_stochastic_layers_draw_during_stack_initialization():
    layered = stack(_warm_noise(), n=3).with_input(jnp.zeros(2))
    key = jax.random.PRNGKey(5)
    first = layered.init(input=jnp.zeros(2), rng=key)
    replay = layered.init(input=jnp.zeros(2), rng=key)
    other = layered.init(input=jnp.zeros(2), rng=jax.random.PRNGKey(6))

    assert layered.contract.init_takes_rng
    assert layered.contract.apply_takes_rng
    assert jnp.allclose(first, replay)
    assert not jnp.allclose(first[1:], other[1:])
    with pytest.raises(TypeError, match='requires rng'):
        layered.init(input=jnp.zeros(2))


def test_primed_stack_initialization_is_a_jax_scan():
    layered = stack(_warm_noise(), n=3).with_input(jnp.zeros(2))
    traced = jax.make_jaxpr(
        lambda key, value: layered.init(input=value, rng=key),
    )(jax.random.PRNGKey(0), jnp.zeros(2))

    assert any(equation.primitive.name == 'scan'
               for equation in traced.jaxpr.eqns)


def test_one_layer_initializes_without_an_unused_apply_draw():
    layered = stack(_warm_noise(), n=1).with_input(jnp.zeros(2))

    state = layered.init(input=jnp.ones(2))

    assert state.shape == (1, 2)
    assert jnp.allclose(state[0], 1.0)


@pytest.mark.parametrize('depth', (0, -1, 1.5))
def test_stack_requires_a_positive_integer_depth(depth):
    with pytest.raises(TypeError, match='positive int'):
        stack(Gain(), n=depth)


def test_stack_layers_are_independent_where_repeat_shares():
    """Verifies that `stack` allocates a leading layer axis on parameters whereas `repeat` keeps parameters unstacked."""
    stacked = stack(Gain(), n=3).parameterize(scale=2.0)
    tied = repeat(Gain(), n=3).parameterize(scale=2.0)

    assert stacked.param.scale.shape == (3,)
    assert tied.param.scale.shape == ()
    assert jnp.allclose(stacked.apply(1.0), tied.apply(1.0))


def test_tiled_layers_own_separate_slots():
    """Verifies that updating one layer's parameter slot in `stack` leaves sibling layers unchanged."""
    stacked = stack(Gain(), n=3).parameterize(scale=2.0)
    assert jnp.allclose(stacked.apply(1.0), 8.0)

    param = stacked.param
    altered = param.replace(scale=param.scale.at[0].set(10.0))
    assert jnp.allclose(stacked.node.bind(altered).apply(1.0), 40.0)

    tied = repeat(Gain(), n=3).parameterize(scale=2.0)
    assert tied.param.scale.shape == ()


def test_a_cyclic_layer_with_no_params_stacks():
    """Verifies that `stack` chains `n` independent state slots for a non-parametric cyclic layer."""
    def Accum():
        def init(node):
            return jnp.zeros_like(node.input)

        def apply(state, input):
            new = state + input
            return new, new

        return Leaf(apply, init=init, name='accum').node

    node = stack(Accum(), n=3).with_input(jnp.zeros(2))
    assert node.param == ()

    state, out = node.apply(node.init(), jnp.ones(2))
    assert jax.tree.leaves(state)[0].shape == (3, 2)
    assert jnp.allclose(out, 1.0)


def test_stack_builds_defaulted_params():
    """Verifies that `stack` broadcasts defaulted and supplied constructor fields to length `n`."""
    from nodejax.control import Integrator

    node = stack(Integrator(), n=2).parameterize()
    assert node.param.decay.shape == (2,)

    node = stack(Integrator(), n=2).parameterize(decay=0.1)
    assert jnp.allclose(node.param.decay, jnp.full(2, 0.1))
    state, out = node.apply(node.init(), 1.0)
    assert jnp.allclose(out, 1.0)
