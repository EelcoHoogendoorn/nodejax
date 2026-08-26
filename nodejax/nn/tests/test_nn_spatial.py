"""Spatial and stochastic stock blocks."""

import jax
import jax.numpy as jnp

from nodejax import batch
from nodejax.nn.conv import Conv, ConvTranspose
from nodejax.nn.spatial import (
    AvgPool, Downsample, GlobalAvgPool, MaxPool, Upsample,
)
from nodejax.nn.stochastic import DropPath, GaussianNoise
from nodejax.struct import Struct


KEY = jax.random.PRNGKey(0)


def test_spatial_pooling_values_and_shapes() -> None:
    image = jnp.arange(16.0).reshape(4, 4, 1)

    maximum = MaxPool().apply(image)
    average = AvgPool().apply(image)
    assert maximum.shape == (2, 2, 1)
    assert jnp.allclose(maximum[..., 0], jnp.array([[5.0, 7.0],
                                                    [13.0, 15.0]]))
    assert jnp.allclose(average[..., 0], jnp.array([[2.5, 4.5],
                                                    [10.5, 12.5]]))
    assert jnp.allclose(GlobalAvgPool().apply(image), jnp.array([7.5]))

    upsampled = Upsample((2, 3), method='nearest').apply(image)
    assert upsampled.shape == (8, 12, 1)
    assert jnp.allclose(upsampled[0, :3], image[0, 0])
    assert Downsample((2, 3)).apply(upsampled).shape == image.shape


def test_same_average_pool_excludes_padding() -> None:
    image = jnp.ones((3, 3, 1))
    output = AvgPool(window=3, stride=2, padding='SAME').apply(image)
    assert output.shape == (2, 2, 1)
    assert jnp.allclose(output, 1.0)


def test_convolution_pooling_and_transpose_compose() -> None:
    input = jnp.ones((8, 8, 2))
    model = (Conv(4) >> MaxPool() >> ConvTranspose(3)).with_input(
        input).parameterize(rng=KEY)

    assert model.param.conv.kernel.shape == (3, 3, 2, 4)
    assert model.param.conv_transpose.kernel.shape == (3, 3, 4, 3)
    assert model.apply(input).shape == (8, 8, 3)

    batched = batch(model.node).bind(model.param)
    output = batched.apply(jnp.ones((5, 8, 8, 2)))
    assert output.shape == (5, 8, 8, 3)


def test_stochastic_blocks_replay_and_map_over_examples() -> None:
    input = Struct(left=jnp.ones(4), right=jnp.ones((2, 3)))
    noise = GaussianNoise(0.2)
    first = noise.apply(input=input, rng=KEY)
    replay = noise.apply(input=input, rng=KEY)
    different = noise.apply(input=input, rng=jax.random.PRNGKey(1))
    assert jnp.allclose(first.left, replay.left)
    assert jnp.allclose(first.right, replay.right)
    assert not jnp.allclose(first.left, different.left)
    assert not jnp.allclose(first.left, first.right.ravel()[:4])

    paths = batch(DropPath(0.5)).apply(
        input=jnp.ones((64, 3)), rng=KEY)
    assert paths.shape == (64, 3)
    assert set(jnp.unique(paths).tolist()) == {0.0, 2.0}


def test_stochastic_eval_builds_are_identity() -> None:
    input = jnp.arange(5.0)
    assert jnp.array_equal(DropPath(0.7, train=False).apply(input), input)
    assert jnp.array_equal(GaussianNoise(2.0, train=False).apply(input), input)
