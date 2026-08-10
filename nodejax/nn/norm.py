"""Normalization layer blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def


@ambient
def LayerNorm(eps: float = 1e-5):
    """Normalize over the feature axis; width from the offer."""
    def param(ndef) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        return Struct(scale=jnp.ones(width), bias=jnp.zeros(width))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)
        return (input - mean) / jnp.sqrt(var + eps) * param.scale + param.bias

    return node_def(apply, param=param, name='norm')


@ambient
def BatchNorm(momentum: float, eps: float = 1e-5, axis: str = 'batch'):
    """Batchnorm written per-sample, nn's own convention: the moments
    are collectives over the NAMED batch axis (lax.pmean over `axis`,
    'batch' by the reserved convention), so the block drops into a
    batch-agnostic pipe as one term and declares the axis need — an
    enclosing batch() binds the name, and binding params or state
    while the need is unmet refuses loudly. Normalizes by the RUNNING
    moments carried as state, then folds the axis moments in; every
    element computes the same pooled moments, so the per-element state
    copies under batch() agree (replicated, never divergent). Eval is
    freezing the state, as ever. Width from the offer; gamma/beta
    restore the affine freedom the normalization removes."""
    def param(ndef) -> Struct:
        width = ndef.apply_input_spec.shape[-1]
        return Struct(gamma=jnp.ones(width), beta=jnp.zeros(width))

    def init(param: Struct) -> Struct:
        return Struct(mean=jnp.zeros_like(param.beta), var=jnp.ones_like(param.gamma))

    def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        out = (input - state.mean) / jnp.sqrt(state.var + eps) * param.gamma + param.beta
        m = jax.lax.pmean(input, axis)
        v = jax.lax.pmean((input - m) ** 2, axis)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     var=(1 - momentum) * state.var + momentum * v)
        return new, out

    return node_def(apply, init=init, param=param, name='bn', tags={'single_batch_state'})


@ambient
def Whiten(momentum: float = 0.1, eps: float = 1e-2, axis: str = 'batch'):
    """ZCA input whitening written per-sample, nn's own convention: the
    moments are collectives over the NAMED batch axis (lax.pmean over
    `axis`, 'batch' by the reserved convention), so an enclosing batch()
    binds the name. Decorrelates by the RUNNING moments carried as state,
    through the inverse matrix square root of the running covariance, then
    folds the axis moments in; every element computes the same pooled
    moments, so the per-element state copies agree (replicated, never
    divergent). Eval is freezing the state, as ever."""
    def init(ndef) -> Struct:
        features = ndef.input
        return Struct(mean=jnp.zeros_like(features),
                      cov=jnp.eye(features.shape[-1], dtype=features.dtype))

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        eigvals, eigvecs = jnp.linalg.eigh(state.cov)
        inv_sqrt = (eigvecs * (1.0 / jnp.sqrt(jnp.maximum(eigvals, 0.0) + eps))) @ eigvecs.T
        whitened = (input - state.mean) @ inv_sqrt

        m = jax.lax.pmean(input, axis)
        centered = input - m
        cov = jax.lax.pmean(jnp.outer(centered, centered), axis)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * m,
                     cov=(1 - momentum) * state.cov + momentum * cov)
        return new, whitened

    return node_def(apply, init=init, name='whiten', tags={'single_batch_state'})
