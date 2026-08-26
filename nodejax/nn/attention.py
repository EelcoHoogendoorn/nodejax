"""Attention and Transformer block primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.compose import serial
from nodejax.transforms.wiring.residual import residual
from nodejax.nn.norm import LayerNorm
from nodejax.nn.mlp import MLP


@node
def tokens():
    """(H, W, C) feature map -> (H*W, C) token sequence."""
    return Leaf(lambda input: input.reshape(-1, input.shape[-1]))


@node(name='pos')
def PosEmbed():
    """Learned position embedding; the whole (T, width) shape from
    the resolved input spec."""
    def param(node, rng) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), node.input.shape))

    def apply(param, input) -> jax.Array:
        return input + param.embed

    return Leaf(apply, param=param)


@node(name='attn')
def Attention(heads: int, causal: bool = False) -> Node:
    """Multi-head self-attention over one `(tokens, width)` sequence.

    Width comes from the resolved input spec and is split evenly over the
    heads. With `causal=True`, each token attends only to itself and earlier
    tokens. The result preserves the input shape.
    """
    def param(node, rng) -> Struct:
        width = node.input.shape[-1]
        if width % heads:
            raise ValueError(f'attention: width {width} not divisible by {heads} heads')
        return Struct(
            wqkv=jax.random.normal(rng.next(), (width, 3 * width)) / jnp.sqrt(width),
            wo=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width))

    def apply(param, input) -> jax.Array:
        dim = input.shape[-1] // heads
        qkv = (input @ param.wqkv).reshape(*input.shape[:-1], 3, heads, dim)
        q, k, v = jnp.unstack(qkv, axis=-3)
        logits = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(dim)
        if causal:
            token_count = input.shape[-2]
            positions = jnp.arange(token_count)
            visible = positions[:, None] >= positions[None, :]
            logits = jnp.where(visible[None, :, :], logits, -jnp.inf)
        mix = jnp.einsum('hqk,khd->qhd', jax.nn.softmax(logits, axis=-1), v)
        return mix.reshape(input.shape) @ param.wo

    return Leaf(apply, param=param)


@node
def TransformerBlock(width: int, heads: int, ratio: int,
                     causal: bool = False) -> Node:
    """Width-preserving pre-norm transformer block.

    `causal` configures the attention member. The block remains suitable for
    `stack` because both residual branches preserve width.
    """
    return serial(
        attn=residual(LayerNorm() >> Attention(heads, causal=causal)),
        mlp=residual(LayerNorm() >> MLP(width, ratio)),
    )
