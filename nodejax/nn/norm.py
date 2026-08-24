"""Normalization layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.node import Node
from nodejax.struct import Struct
from nodejax.ambient import node
from nodejax.authoring import Leaf


@node(name='norm')
def LayerNorm(eps: float = 1e-5):
    r"""Layer normalization over the last (feature) axis.

    Normalizes activations across feature channels and scales with learnable
    parameters `scale` and `bias`:
        y = ((x - mean(x)) / sqrt(var(x) + eps)) * scale + bias

    Parameters
    ----------
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.

    Design Philosophy
    -----------------
    LayerNorm is stateless and parameter-only. Feature width is inferred
    automatically from the resolved input specification (`node.input.shape[-1]`)
    during parameterization, requiring no explicit channel dimension at construction.
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
    """Root mean square normalization over the last axis.

    Feature width comes from the resolved input spec. The learned scale is
    the only parameter; RMS normalization does not center its input or add a
    learned bias.
    """
    def param(node) -> Struct:
        width = node.input.shape[-1]
        return Struct(scale=jnp.ones(width))

    def apply(param, input) -> jax.Array:
        mean_square = jnp.mean(jnp.square(input), axis=-1, keepdims=True)
        return input * jax.lax.rsqrt(mean_square + eps) * param.scale

    return Leaf(apply, param=param)


@node(name='l2norm')
def L2Norm(eps: float = 1e-8) -> Node:
    """Scale vectors on the last axis to unit Euclidean norm."""
    def apply(input) -> jax.Array:
        norm = jnp.linalg.norm(input, axis=-1, keepdims=True)
        return input / (norm + eps)

    return Leaf(apply)


@node
def BatchNorm(momentum: float, eps: float = 1e-5, axis: str = 'batch',
              train: bool = True):
    r"""Batch normalization over a named batch axis with running statistics.

    Normalizes features by running moments and scales with learnable parameters:
        y = ((x - running_mean) / sqrt(running_var + eps)) * gamma + beta

    In training mode (`train=True`), computes batch moments across `axis` using
    `lax.pmean` and updates running statistics via exponential moving average.
    In eval mode (`train=False`), running statistics are held constant.

    Parameters
    ----------
    momentum : float
        EMA factor for updating running mean and variance.
    eps : float, default=1e-5
        Constant added to variance for numerical stability.
    axis : str, default='batch'
        Named collective axis over which batch moments are pooled.
    train : bool, default=True
        Whether to compute batch moments and update running statistics.

    Design Philosophy
    -----------------
    In conventional frameworks, batch normalization is awkward because it mixes
    per-sample computation with cross-sample reductions and mutable mode switches
    like `model.eval()`.

    In NodeJAX:
    - BatchNorm and the `batch()` transform are natural pairs: BatchNorm is authored
      strictly per-sample, declaring its collective reduction over a named axis
      (`axis='batch'`). An enclosing `batch()` transform introduces and binds this
      exact named axis during vectorized execution.
    - Because every sample in the batch computes the same pooled moments via
      `lax.pmean(..., axis)`, the per-sample running state updates remain synchronized
      without needing special-case state containers.
    - `train` is a static constructor argument rather than mutable state.
      Switching a trained model to evaluation mode is done functionally by
      rebuilding the tree via `model.specialize(**{'*.train': False})`.
    """
    def param(node) -> Struct:
        width = node.input.shape[-1]
        return Struct(gamma=jnp.ones(width), beta=jnp.zeros(width))

    def init(param) -> Struct:
        return Struct(mean=jnp.zeros_like(param.beta), var=jnp.ones_like(param.gamma))

    def apply(param, state, input) -> tuple[Struct, jax.Array]:
        out = (input - state.mean) / jnp.sqrt(state.var + eps) * param.gamma + param.beta
        if not train:                       # eval build: read, never write
            return state, out
        m = jax.lax.pmean(input, axis)
        v = jax.lax.pmean((input - m) ** 2, axis)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     var=(1 - momentum) * state.var + momentum * v)
        return new, out

    return Leaf(apply, init=init, param=param,
                    tags={'single_batch_state', 'running_stats'})


@node
def Whiten(momentum: float = 0.1, eps: float = 1e-2, axis: str = 'batch',
           train: bool = True):
    r"""ZCA input whitening over a named batch axis with running statistics.

    Decorrelates features using the running mean and inverse matrix square root
    of the running covariance:
        y = (x - running_mean) @ inv_sqrt_cov

    In training mode (`train=True`), computes batch covariance across `axis` using
    `lax.pmean` and updates running mean and covariance via exponential moving average.
    In eval mode (`train=False`), running statistics are held constant.

    Parameters
    ----------
    momentum : float, default=0.1
        EMA factor for updating running mean and covariance.
    eps : float, default=1e-2
        Small ridge regularization constant added to eigenvalues before inversion.
    axis : str, default='batch'
        Named collective axis over which batch moments are pooled.
    train : bool, default=True
        Whether to compute batch moments and update running statistics.

    Design Philosophy
    -----------------
    Like `BatchNorm`, `Whiten` is authored strictly per-sample and paired with
    the `batch()` transform. Batch covariance is computed collectively via
    `lax.pmean` across `axis='batch'`, updating pure functional running state
    identically across the batch without mutable global state.
    """
    def init(node) -> Struct:
        features = node.input
        return Struct(mean=jnp.zeros_like(features),
                      cov=jnp.eye(features.shape[-1], dtype=features.dtype))

    def apply(state, input) -> tuple[Struct, jax.Array]:
        eigvals, eigvecs = jnp.linalg.eigh(state.cov)
        inv_sqrt = (eigvecs * (1.0 / jnp.sqrt(jnp.maximum(eigvals, 0.0) + eps))) @ eigvecs.T
        whitened = (input - state.mean) @ inv_sqrt

        if not train:                       # eval build: read, never write
            return state, whitened
        m = jax.lax.pmean(input, axis)
        centered = input - m
        cov = jax.lax.pmean(jnp.outer(centered, centered), axis)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     cov=(1 - momentum) * state.cov + momentum * cov)
        return new, whitened

    return Leaf(apply, init=init,
                    tags={'single_batch_state', 'running_stats'})
