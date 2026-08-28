"""Pure stock activation and encoding blocks."""

import jax
import jax.numpy as jnp

from nodejax import ensemble, nn


def test_activation_blocks_match_jax() -> None:
    input = jnp.array([-2.0, 0.0, 2.0])
    pairs = (
        (nn.elu, jax.nn.elu),
        (nn.leaky_relu, jax.nn.leaky_relu),
        (nn.softplus, jax.nn.softplus),
        (nn.softmax, jax.nn.softmax),
        (nn.log_softmax, jax.nn.log_softmax),
    )
    for block, function in pairs:
        assert jnp.allclose(block.apply(input), function(input))
    assert jnp.array_equal(nn.identity.apply(input), input)


def test_one_hot_composes_with_linear() -> None:
    input = jnp.array(2)
    model = (nn.OneHot(5) >> nn.Linear(3)).with_input(
        input).parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.linear.w.shape == (5, 3)
    assert model.apply(input).shape == (3,)


def test_projection_removes_the_feature_axis_and_ensemble_adds_one() -> None:
    key = jax.random.PRNGKey(0)
    projection = nn.Projection().with_input(jnp.zeros(4)).parameterize(rng=key)
    linear = ensemble(nn.Projection(), n=3).with_input(
        jnp.zeros(4)).parameterize(rng=key)

    assert projection.param.w.shape == (4,)
    assert projection.apply(jnp.ones(4)).shape == ()
    assert linear.param.w.shape == (3, 4)
    assert linear.apply(jnp.ones(4)).shape == (3,)


def test_linear_and_projection_accept_parameter_initializers() -> None:
    key = jax.random.PRNGKey(0)
    linear = nn.Linear(
        3,
        weight_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.ones,
    ).with_input(jnp.zeros(4)).parameterize(rng=key)
    projection = nn.Projection(
        weight_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.ones,
    ).with_input(jnp.zeros(4)).parameterize(rng=key)

    assert jnp.array_equal(linear.param.w, jnp.zeros((4, 3)))
    assert jnp.array_equal(linear.param.b, jnp.ones(3))
    assert jnp.array_equal(projection.param.w, jnp.zeros(4))
    assert jnp.array_equal(projection.param.b, jnp.ones(()))
