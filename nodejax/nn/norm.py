"""Normalization layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


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


def _reduction_axes(axis) -> tuple[tuple, tuple]:
    """Split a reduction spec into collective names and positional axes."""
    axes = axis if type(axis) is tuple else (axis,)
    return (tuple(a for a in axes if type(a) is str),
            tuple(a for a in axes if type(a) is int))


def _leading_axes(positional: tuple, sample: tuple, who: str) -> tuple:
    """Resolve positional reduction axes; only leading sample axes qualify."""
    rank = len(sample)
    resolved = tuple(a if a >= 0 else a + rank for a in positional)
    for index in resolved:
        if not 0 <= index < rank - 1:
            raise ValueError(
                f'{who}: positional reduction axis {index} of sample shape '
                f'{sample} is the last axis, which the running statistics '
                'are laid out over; only leading axes pool')
    return resolved


def _statistics_shape(sample: tuple, reduced: tuple) -> tuple:
    """The sample shape with the reduced axes kept at extent one."""
    kept = set(reduced)
    return tuple(1 if index in kept else extent
                 for index, extent in enumerate(sample))


@node
def BatchNorm(momentum: float, eps: float = 1e-5,
              axis: str | int | tuple = 'batch',
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
    axis : str, int, or tuple of these, default='batch'
        Axes over which batch moments are pooled. A name reaches across the
        enclosing map that binds it; an int selects a leading axis of the
        sample itself.
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

    Statistics reduce over exactly the listed axes. A name reaches across
    the enclosing map that binds it; an int selects a leading axis of the
    sample itself and stays in the running statistics at extent one. What
    is not listed is not pooled: unlisted leading axes keep one running
    moment per position, with state bound to the extents it was
    initialized for. Naming an axis the moments do not pool fails loudly
    at trace time: the shared running state is only consistent across
    pooled axes.

    Over sequences, for example, these spellings are the published
    treatments: per-position statistics under a plain `batch()` is
    recurrent batch normalization (Cooijmans et al. 2017), while listing
    the time axis, `axis=('batch', 0)`, pools it and matches torch's
    BatchNorm1d over (N, C, L), as does binding it to a name with an inner
    `batch(norm, axis='stream')` and listing that name.
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
           axis: str | int | tuple = 'batch',
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
    axis : str, int, or tuple of these, default='batch'
        Axes over which batch moments are pooled. A name reaches across the
        enclosing map that binds it; an int selects a leading axis of the
        sample itself.
    train : bool, default=True
        Whether to compute batch moments and update running statistics.

    Design Philosophy
    -----------------
    Like `BatchNorm`, `Whiten` is authored strictly per-sample and paired with
    the `batch()` transform. Batch covariance is computed collectively via
    `lax.pmean` across `axis='batch'`, updating pure functional running state
    identically across the batch without mutable global state.

    As with `BatchNorm`, moments reduce over exactly the listed axes: a
    name reaches across the enclosing map that binds it, an int selects a
    leading axis of the sample itself and stays in the running moments at
    extent one, and unlisted leading axes keep one moment per position.
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
