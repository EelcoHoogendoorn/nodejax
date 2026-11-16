"""Defines multi-head self-attention, position embeddings, and pre-norm Transformer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.compose import serial
from nodejax.transforms.wiring.residual import residual
from nodejax.nn.norm import LayerNorm
from nodejax.nn.mlp import MLP


@node
def tokens() -> PNode:
    """Flattens spatial axes of a `(H, W, C)` feature map into a `(H * W, C)` token sequence."""
    return Leaf(lambda input: input.reshape(-1, input.shape[-1]))


@node(name='pos')
def PosEmbed() -> Node:
    """Adds a learned `(tokens, width)` position embedding sized from `node.input.shape`."""
    def param(node, rng) -> Struct:
        return Struct(embed=0.02 * jax.random.normal(rng.next(), node.input.shape))

    def apply(param, input) -> jax.Array:
        return input + param.embed

    return Leaf(apply, param=param)


@node(name='attn')
def Attention(heads: int, causal: bool = False) -> Node:
    """Computes multi-head self-attention over one `(tokens, width)` sequence.

    Splits `node.input.shape[-1]` evenly across `heads` and preserves the input shape. Setting `causal=True` masks positions ahead of each query token.
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
    """Applies pre-norm multi-head self-attention followed by a pre-norm MLP, each wrapped in a residual skip.

    Both residual branches preserve `width`, making the block directly compatible with `stack` and `repeat`.
    """
    return serial(
        attn=residual(LayerNorm() >> Attention(heads, causal=causal)),
        mlp=residual(LayerNorm() >> MLP(width, ratio)),
    )
