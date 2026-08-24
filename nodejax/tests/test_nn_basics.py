"""Pure stock activation and encoding blocks."""

import jax
import jax.numpy as jnp

from nodejax import nn


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
