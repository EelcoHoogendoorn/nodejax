"""Defines 2D spatial pooling and resizing Leaf factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.pnode import PNode
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return value


@node
def MaxPool(window: int | tuple[int, int] = 2,
            stride: int | tuple[int, int] | None = None,
            padding: str = 'VALID') -> PNode:
    """Computes sliding-window maxima over one `(height, width, channels)` feature map."""
    window_shape = _pair(window)
    stride_shape = _pair(window if stride is None else stride)

    def apply(input) -> jax.Array:
        return jax.lax.reduce_window(
            input,
            -jnp.inf,
            jax.lax.max,
            window_shape + (1,),
            stride_shape + (1,),
            padding.upper(),
        )

    return Leaf(apply)


@node
def AvgPool(window: int | tuple[int, int] = 2,
            stride: int | tuple[int, int] | None = None,
            padding: str = 'VALID') -> PNode:
    """Computes sliding-window means over one `(height, width, channels)` feature map, excluding padded positions."""
    window_shape = _pair(window)
    stride_shape = _pair(window if stride is None else stride)

    def apply(input) -> jax.Array:
        dimensions = window_shape + (1,)
        strides = stride_shape + (1,)
        total = jax.lax.reduce_window(
            input, 0.0, jax.lax.add, dimensions, strides, padding.upper())
        count = jax.lax.reduce_window(
            jnp.ones(input.shape[:2] + (1,), dtype=input.dtype),
            0.0,
            jax.lax.add,
            dimensions,
            strides,
            padding.upper(),
        )
        return total / count

    return Leaf(apply)


@node
def GlobalAvgPool() -> PNode:
    """Averages the two leading spatial axes of one `(height, width, channels)` feature map."""
    return Leaf(lambda input: jnp.mean(input, axis=(0, 1)))


@node
def Upsample(scale: int | tuple[int, int] = 2,
             method: str = 'nearest') -> PNode:
    """Multiplies the spatial dimensions of one `(height, width, channels)` feature map by `scale`."""
    scale_shape = _pair(scale)
    if scale_shape[0] < 1 or scale_shape[1] < 1:
        raise ValueError('upsample scale must be positive')

    def apply(input) -> jax.Array:
        height, width, channels = input.shape
        shape = (height * scale_shape[0], width * scale_shape[1], channels)
        return jax.image.resize(input, shape, method=method)

    return Leaf(apply)


@node
def Downsample(scale: int | tuple[int, int] = 2,
               method: str = 'nearest') -> PNode:
    """Divides the spatial dimensions of one `(height, width, channels)` feature map by integer `scale`."""
    scale_shape = _pair(scale)
    if scale_shape[0] < 1 or scale_shape[1] < 1:
        raise ValueError('downsample scale must be positive')

    def apply(input) -> jax.Array:
        height, width, channels = input.shape
        shape = (height // scale_shape[0], width // scale_shape[1], channels)
        return jax.image.resize(input, shape, method=method)

    return Leaf(apply)
