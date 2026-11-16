"""Homogeneous depth composition (`stack`) compiled as a single `jax.lax.scan`.

`stack` composes `n` identically structured layers with independent parameter
and state slices stacked along leading axis 0 so XLA compiles the entire depth
stack once instead of unrolling `n` separate Python layer graphs.
"""

from __future__ import annotations

from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.iteration._common import scanned_apply, scanned_initialize
from nodejax.transforms.transform import transform, vmap_init, vmap_param


@transform
def stack(layer: Node, n: int) -> Node:
    """Chains `n` independently parameterized copies of `layer` sequentially via `jax.lax.scan`.

    Unlike `repeat`, which shares one parameter tree across `n` steps, `stack`
    gives parameter and state trees a leading axis of length `n` while keeping
    external input and output shapes equal to a single `layer`. Calling
    `parameterize(...)` tiles explicit overrides across all `n` layers and
    splits PRNG keys per layer, while `stack(layer, n).bind(param)` attaches an
    already-stacked parameter tree directly.
    """
    if type(n) is not int or n < 1:
        raise TypeError(f'stack depth must be a positive int, got {n!r}')
    def apply_fn(contract, param, state, input, rng):
        return scanned_apply(
            contract.members.layer,
            param, state, input, rng,
            scanned_params=True, length=n)

    def param_fn(contract, param_input, rng):
        return vmap_param(
            contract.members.layer, contract,
            rng, param_input, count=n)

    def init_fn(contract, param, state_input, rng):
        inner = contract.members.layer
        return vmap_init(
            inner,
            contract,
            rng,
            param,
            state_input,
            count=n,
            param_axis=0 if inner.parametric else None,
        )

    def prime_fn(contract, param, state_input, input, rng):
        inner = contract.members.layer
        return scanned_initialize(
            inner, param, state_input, input, rng,
            count=n, scanned_params=inner.parametric)

    return Wrapper(layer=layer).roles(
        destructurable=False,
        name=f'stack({layer.name})',
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        init_takes_rng=(True if (
            layer.contract.init_requires_input
            and n > 1 and layer.contract.apply_takes_rng
        ) else None),
    )
