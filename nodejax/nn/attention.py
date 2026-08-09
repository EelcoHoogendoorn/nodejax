"""Attention and Transformer block primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream
from nodejax.compose import serial
from nodejax.transforms.residual import residual
from nodejax.nn.norm import LayerNorm
from nodejax.nn.mlp import MLP


def tokens():
    """(H, W, C) feature map -> (H*W, C) token sequence."""
    return node_def(lambda input: input.reshape(-1, input.shape[-1]), name='tokens')


@ambient
def PosEmbed():
    """Learned position embedding; the whole (T, width) shape from
    the offer."""
    def param(ndef, rng: KeyStream) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), ndef.apply_input_spec.shape))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input + param.embed

    return node_def(apply, param=param, name='pos')


@ambient
def Attention(heads: int):
    """Multi-head self-attention over one (T, width) sequence,
    width-preserving; width from the offer, split over heads."""
    def param(ndef, rng: KeyStream) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        if width % heads:
            raise ValueError(f'attention: width {width} not divisible by {heads} heads')
        return Struct(
            wqkv=jax.random.normal(rng.next(), (width, 3 * width)) / jnp.sqrt(width),
            wo=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        dim = input.shape[-1] // heads
        qkv = (input @ param.wqkv).reshape(*input.shape[:-1], 3, heads, dim)
        q, k, v = jnp.unstack(qkv, axis=-3)
        logits = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(dim)
        mix = jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), v)
        return mix.reshape(input.shape) @ param.wo

    return node_def(apply, param=param, name='attn')


@ambient
def Block(width: int, heads: int, ratio: int):
    """Pre-norm transformer block at an explicit width; width-preserving,
    so stack-able."""
    return serial(
        attn=residual(LayerNorm() >> Attention(heads)),
        mlp=residual(LayerNorm() >> MLP(width, ratio)),
    )
