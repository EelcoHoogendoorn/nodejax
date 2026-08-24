"""stack: scan over the layer axis, per-layer params, layer k feeds k+1.

The distinction that gives stack its meaning is against `repeat`: stack gives
every layer its OWN params, repeat threads ONE set through n applications.
Everything below is about honouring that on both sides of the contract, since
the param side and the state side each have to believe the same depth.
"""

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


# --- how the depth is supplied ---

def test_per_layer_values_are_a_param_tree_not_a_construction():
    """n says the depth, always. Finished per-layer values are not something
    to construct FROM, they are the param tree itself, so they arrive by
    bind rather than by parameterize."""
    s = stack(Gain(), n=3).bind(Struct(scale=jnp.array([2.0, 3.0, 4.0])))
    assert s.param.scale.shape == (3,)
    assert jnp.allclose(s.apply(1.0), 24.0)


def test_n_draws_a_layer_per_key():
    """With n and an rng, each layer is an independent draw."""
    s = stack(nn.Linear(4), n=3).with_input(jnp.zeros(4)).parameterize(
        rng=jax.random.PRNGKey(0))
    assert s.param.w.shape == (3, 4, 4)
    assert not jnp.allclose(s.param.w[0], s.param.w[1])   # independent, not copied


def test_n_tiles_a_value_the_caller_gives_once():
    """With n and a VALUE, the caller supplies one layer's worth and gets n of
    them. They start equal, which is not the same as being shared: the layers
    are independent and diverge under training."""
    s = stack(Gain(), n=3).parameterize(scale=2.0)
    assert s.param.scale.shape == (3,)
    assert jnp.allclose(s.param.scale, 2.0)
    assert jnp.allclose(s.apply(1.0), 8.0)                # 2 * 2 * 2


def test_n_tiles_the_defaults_too():
    """A param left at its default is supplied by nobody, and still has to
    reach every layer. This is the case that fails loudest, because nothing in
    the bundle carries a depth for the state side to agree with."""
    s = scanned(stack(Integrator(), n=2)).parameterize()
    assert s.param.decay.shape == (2,)
    assert jnp.allclose(s.apply(jnp.ones(3)), jnp.array([1.0, 3.0, 6.0]))


def test_the_state_side_agrees_with_the_depth():
    """Both halves of the contract must believe the same n. When they disagree
    the failure is a vmap rank error naming neither the leaf nor n."""
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


# --- stack against repeat: independent versus shared ---

def test_stack_layers_are_independent_where_repeat_shares():
    """The defining difference, structurally."""
    stacked = stack(Gain(), n=3).parameterize(scale=2.0)
    tied = repeat(Gain(), n=3).parameterize(scale=2.0)

    assert stacked.param.scale.shape == (3,)      # one per layer
    assert tied.param.scale.shape == ()           # one for all of them
    assert jnp.allclose(stacked.apply(1.0), tied.apply(1.0))   # same computation, though


def test_tiled_layers_own_separate_slots():
    """Starting equal is not being shared. Each layer owns its slot, so writing
    one changes that layer alone.

    Note that TRAINING will not separate them here, and that is arithmetic
    rather than sharing: three gains in series with equal scales have identical
    gradients by symmetry, d/ds1 = 2(p-1)*s2*s3 = d/ds2. Independence is a
    property of the parameterization, not of whether a particular objective
    happens to exploit it."""
    stacked = stack(Gain(), n=3).parameterize(scale=2.0)
    assert jnp.allclose(stacked.apply(1.0), 8.0)

    param = stacked.param
    altered = param.replace(scale=param.scale.at[0].set(10.0))
    assert jnp.allclose(stacked.node.bind(altered).apply(1.0), 40.0)   # 10 * 2 * 2

    tied = repeat(Gain(), n=3).parameterize(scale=2.0)
    assert tied.param.scale.shape == ()          # nothing to write one layer of


def test_a_cyclic_layer_with_no_params_stacks():
    """stack's own guard admits `parametric OR cyclic`, so a cyclic node with
    no params is a supported node and not a degenerate one: three layers, each
    with its own state, chained. It used to raise IndexError from deep inside
    the scan, because the depth was counted off the param rows and there were
    none. The depth is a construction argument; nothing needs counting."""
    def Accum():
        def init(node):
            return jnp.zeros_like(node.input)

        def apply(state, input):
            new = state + input
            return new, new

        return Leaf(apply, init=init, name='accum').node

    node = stack(Accum(), n=3).with_input(jnp.zeros(2))
    assert node.param == ()                          # nothing to count

    state, out = node.apply(node.init(), jnp.ones(2))
    assert jax.tree.leaves(state)[0].shape == (3, 2)  # one state per layer
    assert jnp.allclose(out, 1.0)                     # each layer adds its own carry


def test_stack_builds_defaulted_params():
    """Defaults stack: a field left at its default broadcasts across the
    layer axis beside its supplied siblings, one row per layer either way.
    (An Integrator docstring once filed this as a known limitation; the
    axis_size in vmapped_parameterize is the fix, and this is its pin.)"""
    from nodejax.control import Integrator

    node = stack(Integrator(), n=2).parameterize()          # all defaulted
    assert node.param.decay.shape == (2,)

    node = stack(Integrator(), n=2).parameterize(decay=0.1)  # scalar supplied
    assert jnp.allclose(node.param.decay, jnp.full(2, 0.1))
    state, out = node.apply(node.init(), 1.0)
    assert jnp.allclose(out, 1.0)
