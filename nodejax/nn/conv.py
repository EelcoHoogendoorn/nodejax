"""Convolutional layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.core.node import Node
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def Conv(features: int, kernel: int = 3, stride: int = 1) -> Node:
    """SAME convolution over one (H, W, C) map; in-channels from the
    resolved input spec. lax demands a batch dim, so a unit one is added and
    stripped; under batch() the vmap fuses it into a real batched
    conv."""
    def param(node, rng) -> Struct:
        c_in = node.input.shape[-1]
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, features))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in),
                      bias=jnp.zeros(features))

    def apply(param, input) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(stride, stride),
            padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return out + param.bias

    return Leaf(apply, param=param)


@node
def ConvTranspose(features: int, kernel: int = 3, stride: int = 2) -> Node:
    """SAME transposed convolution over one feature map.

    Input channels come from the resolved input spec. A stride greater than
    one expands the spatial axes by that factor.
    """
    def param(node, rng) -> Struct:
        input_channels = node.input.shape[-1]
        weight = jax.random.normal(
            rng.next(), (kernel, kernel, input_channels, features))
        return Struct(
            kernel=weight / jnp.sqrt(kernel * kernel * input_channels),
            bias=jnp.zeros(features),
        )

    def apply(param, input) -> jax.Array:
        output = jax.lax.conv_transpose(
            input[None],
            param.kernel,
            strides=(stride, stride),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )[0]
        return output + param.bias

    return Leaf(apply, param=param)
