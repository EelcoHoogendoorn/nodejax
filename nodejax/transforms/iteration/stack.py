from __future__ import annotations

from nodejax.core.node import Node
from nodejax.transforms.transform import transform, vmap_init, vmap_param
from nodejax.transforms.iteration._common import scanned_apply, scanned_initialize
from nodejax.core.wrapper import Wrapper


@transform
def stack(layer: Node, n: int) -> Node:
    """Compose ``n`` independently parameterized copies of ``layer``.

    Each layer receives the previous layer's output. Parameters and state, when
    present, gain a leading axis of length ``n``; the external input and output
    retain the original layer's shape. A node with neither is still valid and
    simply has no per-layer values to store.

    Arguments passed to ``parameterize`` are shared across the layers, while
    each randomized parameter initialization receives its own key. To use an
    existing set of per-layer parameters instead, bind the stacked parameter
    tree directly with ``stack(layer, n).bind(param)``.
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
