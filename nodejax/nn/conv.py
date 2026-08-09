"""Convolutional layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream


@ambient
def Conv(features: int, kernel: int = 3, stride: int = 1):
    """SAME convolution over one (H, W, C) map; in-channels from the
    offer. lax demands a batch dim, so a unit one is added and
    stripped; under batch() the vmap fuses it into a real batched
    conv."""
    def param(ndef, rng: KeyStream) -> Struct:
        c_in = ndef.apply_input_spec.shape[-1]
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, features))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in),
                      bias=jnp.zeros(features))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(stride, stride),
            padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return out + param.bias

    return node_def(apply, param=param, name='conv')

