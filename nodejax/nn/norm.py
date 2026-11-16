"""Defines feature normalization and ZCA whitening Leaf factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def LayerNorm(eps: float = 1e-5) -> Node:
    """Normalizes activations across the trailing feature axis with learned `scale` and `bias`.

    Infers the feature width from `node.input.shape[-1]` and computes `((input - mean) / sqrt(var + eps)) * scale + bias`.
    """
    def param(node) -> Struct:
        width = node.input.shape[-1]
        return Struct(scale=jnp.ones(width), bias=jnp.zeros(width))

    def apply(param, input) -> jax.Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)
        return (input - mean) / jnp.sqrt(var + eps) * param.scale + param.bias

    return Leaf(apply, param=param)


@node
def RMSNorm(eps: float = 1e-6) -> Node:
    """Scales activations by their root mean square along the trailing axis with a learned `scale`.

    Infers the feature width from `node.input.shape[-1]` without centering or adding a bias term.
    """
    def param(node) -> Struct:
        width = node.input.shape[-1]
        return Struct(scale=jnp.ones(width))

    def apply(param, input) -> jax.Array:
        mean_square = jnp.mean(jnp.square(input), axis=-1, keepdims=True)
        return input * jax.lax.rsqrt(mean_square + eps) * param.scale

    return Leaf(apply, param=param)


@node
def L2Norm(eps: float = 1e-8) -> PNode:
    """Divides vectors along the trailing axis by their Euclidean norm plus `eps`."""
    def apply(input) -> jax.Array:
        norm = jnp.linalg.norm(input, axis=-1, keepdims=True)
        return input / (norm + eps)

    return Leaf(apply)


def _reduction_axes(
        axis: str | int | tuple[str | int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Splits an axis specification into collective axis names and positional sample axes."""
    axes = axis if type(axis) is tuple else (axis,)
    return (tuple(a for a in axes if type(a) is str),
            tuple(a for a in axes if type(a) is int))


def _leading_axes(
        positional: tuple[int, ...], sample: tuple[int, ...], who: str,
) -> tuple[int, ...]:
    """Resolves positional reduction axes against `sample` and rejects reducing the trailing feature axis."""
    rank = len(sample)
    resolved = tuple(a if a >= 0 else a + rank for a in positional)
    for index in resolved:
        if not 0 <= index < rank - 1:
            raise ValueError(
                f'{who}: positional reduction axis {index} of sample shape '
                f'{sample} is the last axis, which the running statistics '
                'are laid out over; only leading axes pool')
    return resolved


def _statistics_shape(
        sample: tuple[int, ...], reduced: tuple[int, ...],
) -> tuple[int, ...]:
    """Replaces each axis index in `reduced` with extent `1` inside `sample`."""
    kept = set(reduced)
    return tuple(1 if index in kept else extent
                 for index, extent in enumerate(sample))


@node
def BatchNorm(momentum: float, eps: float = 1e-5,
              axis: str | int | tuple[str | int, ...] = 'batch',
              train: bool = True) -> Node:
    """Normalizes per-sample activations by running moments pooled over `axis` and scales by `gamma` and `beta`.

    Authored per sample with tags `'single_batch_state'` and `'running_stats'`. String entries in `axis` pool across enclosing `batch` maps via `jax.lax.pmean`; integer entries pool leading positional axes of the sample and keep them at extent `1` in the running state, while unlisted leading axes keep separate statistics per position. When `train=True`, updates `state.mean` and `state.var` with exponential factor `momentum`; when `train=False`, reads the stored moments without updating them.
    """
    def param(node) -> Struct:
        width = node.input.shape[-1]
        return Struct(gamma=jnp.ones(width), beta=jnp.zeros(width))

    names, positional = _reduction_axes(axis)

    def init(node, param) -> Struct:
        sample = node.input.shape
        kept = _statistics_shape(
            sample, _leading_axes(positional, sample, 'batch_norm'))
        return Struct(mean=jnp.zeros(kept, dtype=param.beta.dtype),
                      var=jnp.ones(kept, dtype=param.gamma.dtype))

    def apply(param, state, input) -> tuple[Struct, jax.Array]:
        out = (input - state.mean) / jnp.sqrt(state.var + eps) * param.gamma + param.beta
        if not train:                       # eval build: read, never write
            return state, out
        reduced = _leading_axes(positional, input.shape, 'batch_norm')
        m = jnp.mean(input, reduced, keepdims=True)
        if names:
            m = jax.lax.pmean(m, names)
        v = jnp.mean((input - m) ** 2, reduced, keepdims=True)
        if names:
            v = jax.lax.pmean(v, names)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     var=(1 - momentum) * state.var + momentum * v)
        return new, out

    return Leaf(apply, init=init, param=param,
                    tags={'single_batch_state', 'running_stats'})


@node
def Whiten(momentum: float = 0.1, eps: float = 1e-2,
           axis: str | int | tuple[str | int, ...] = 'batch',
           train: bool = True) -> PNode:
    """Decorrelates trailing features via ZCA whitening `(input - mean) @ inv_sqrt_cov` over `axis`.

    Authored per sample with tags `'single_batch_state'` and `'running_stats'`, following the same named and positional reduction rules as `BatchNorm`. Regularizes eigenvalues of `state.cov` by `eps` before inversion, updates `state.mean` and `state.cov` via `jax.lax.pmean` with factor `momentum` when `train=True`, and holds running statistics fixed when `train=False`.
    """
    names, positional = _reduction_axes(axis)

    def init(node) -> Struct:
        features = node.input
        sample = features.shape
        width = sample[-1]
        kept = _statistics_shape(
            sample, _leading_axes(positional, sample, 'whiten'))
        eye = jnp.eye(width, dtype=features.dtype)
        return Struct(mean=jnp.zeros(kept, dtype=features.dtype),
                      cov=jnp.broadcast_to(eye, kept[:-1] + (width, width)))

    def apply(state, input) -> tuple[Struct, jax.Array]:
        eigvals, eigvecs = jnp.linalg.eigh(state.cov)
        scale = 1.0 / jnp.sqrt(jnp.maximum(eigvals, 0.0) + eps)
        inv_sqrt = (eigvecs * scale[..., None, :]) @ jnp.swapaxes(eigvecs, -1, -2)
        whitened = jnp.einsum('...i,...ij->...j', input - state.mean, inv_sqrt)

        if not train:                       # eval build: read, never write
            return state, whitened
        reduced = _leading_axes(positional, input.shape, 'whiten')
        m = jnp.mean(input, reduced, keepdims=True)
        if names:
            m = jax.lax.pmean(m, names)
        centered = input - m
        cov = jnp.mean(
            jnp.einsum('...i,...j->...ij', centered, centered),
            reduced, keepdims=True)
        if names:
            cov = jax.lax.pmean(cov, names)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     cov=(1 - momentum) * state.cov + momentum * cov)
        return new, whitened

    return Leaf(apply, init=init,
                    tags={'single_batch_state', 'running_stats'})
