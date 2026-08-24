from __future__ import annotations

from nodejax.ambient import node
from nodejax.binding import Aux
from nodejax.node import Node
from nodejax.pnode import PNode
from nodejax.transform import (
    transform, vmap_apply, vmap_init, vmap_prime, vmap_param,
)
from nodejax.wrapper import Wrapper


@transform
def ensemble(member: Node, n: int,
             axis: str = 'ensemble') -> Node:
    """Build ``n`` independent copies along one named axis.

    Param constructs one parameter row per member. Init constructs or primes
    one state row per member. Apply broadcasts runtime input and stacks output.
    A stochastic role splits its RNG per member. The member's input forms,
    priming requirement, and apply shape evidence remain unchanged.
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
def reduce(fn) -> PNode:
    """Reduce the leading ensemble axis and retain the rows as auxiliary data."""
    from nodejax.authoring import Leaf

    def apply(input):
        return fn(input, axis=0), Aux(population=input)

    return Leaf(apply, name=f"reduce({getattr(fn, '__name__', 'fn')})")
