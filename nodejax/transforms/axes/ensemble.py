"""Population-axis replication (`ensemble`) and reduction (`reduce`).

`ensemble` replicates a parametric or cyclic `Node` into `n` independent copies
along leading axis 0, broadcasting each runtime input across all copies and
stacking their outputs. `reduce` collapses that leading population axis with an
aggregation function such as `jnp.mean` while sowing the unreduced rows under
`Aux(population=...)`.
"""

from __future__ import annotations

from typing import Callable

from nodejax.core.ambient import node
from nodejax.core.binding import Aux
from nodejax.core.node import Node
from nodejax.core.pnode import PNode
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.transform import (
    transform, vmap_apply, vmap_init, vmap_param, vmap_prime,
)


@transform
def ensemble(member: Node, n: int,
             axis: str = 'ensemble') -> Node:
    """Replicates `member` into `n` independent parameter and state rows along named axis `axis`.

    Unlike `batch`, `ensemble` allocates `n` distinct parameter and state slices
    at leading axis 0, broadcasts runtime inputs across all `n` copies, splits
    PRNG streams per copy, and stacks their outputs along axis 0.
    """
    if not (member.parametric or member.cyclic):
        raise TypeError(f'ensemble of {member!r}: no params and no state means no '
                        'member identity; use the node directly')
    num_members = n

    def apply_fn(contract, param, state, input, rng):
        inner = contract.members.member
        return vmap_apply(
            inner, param, state, input, rng,
            param_axis=0,
            state_axis=0,
            input_axis=None,
            axis_name=axis,
            count=num_members,
        )

    def param_fn(contract, param_input, rng):
        return vmap_param(
            contract.members.member, contract,
            rng, param_input, count=num_members)

    def init_fn(contract, param, state_input, rng):
        inner = contract.members.member
        return vmap_init(
            inner, contract, rng, param, state_input,
            count=num_members,
            param_axis=0)

    def prime_fn(contract, param, state_input, input, rng):
        inner = contract.members.member
        return vmap_prime(
            inner, param, state_input, input, rng,
            count=num_members,
            param_axis=0,
            input_axis=None,
            state_axis=0,
        )

    return Wrapper(member=member).roles(
        destructurable=False,
        name=f'ensemble({member.name})',
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
    )


@node
def reduce(fn: Callable) -> PNode:
    """Aggregates leading axis 0 via `fn(input, axis=0)` and sows `Aux(population=input)`."""
    from nodejax.core.authoring import Leaf

    def apply(input):
        return fn(input, axis=0), Aux(population=input)

    return Leaf(apply, name=f"reduce({getattr(fn, '__name__', 'fn')})")
