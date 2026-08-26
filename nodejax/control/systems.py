"""Standard dynamic plant models."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.core.types import PyTree


@node
def FirstOrder(dt: float, warm: bool = False) -> Node:
    """First-order lag discretized with backward Euler.

    `tau` is the time constant and `gain` is the steady-state gain. Cold
    initialization starts the output at zero. With `warm=True`, init requires
    an input value and starts at the corresponding steady-state output.
    """
    if dt <= 0.0:
        raise ValueError('dt must be positive')

    def param(tau: PyTree, gain: PyTree = 1.0) -> Struct:
        return Struct(tau=jnp.asarray(tau), gain=jnp.asarray(gain))

    def target(param, input) -> PyTree:
        return jax.tree.map(lambda value: param.gain * value, input)

    def init(param, input) -> PyTree:
        return target(param, input)

    def init_cold(node, param) -> PyTree:
        zeros = jax.tree.map(jnp.zeros_like, node.input)
        return target(param, zeros)

    def apply(param, state, input) -> tuple[PyTree, PyTree]:
        alpha = dt / (param.tau + dt)
        desired = target(param, input)
        state = jax.tree.map(
            lambda value, goal: value + alpha * (goal - value),
            state,
            desired,
        )
        return state, state

    return Leaf(apply, param=param, init=init if warm else init_cold)


@node
def StateSpace() -> Node:
    """Discrete linear state-space system.

    The transition and output equations are `x_next = A @ x + B @ input`
    and `output = C @ x + D @ input`. Output is computed from the current
    state before the transition. State initializes to zero; bind a different
    state explicitly when a model starts from another operating point.
    """
    def param(A: PyTree, B: PyTree, C: PyTree, D: PyTree) -> Struct:
        A = jnp.asarray(A)
        B = jnp.asarray(B)
        C = jnp.asarray(C)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('A must be a square matrix')
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError('B must have one row per state')
        if C.ndim != 2 or C.shape[1] != A.shape[0]:
            raise ValueError('C must have one column per state')

        D = jnp.asarray(D)
        if D.ndim != 2 or D.shape != (C.shape[0], B.shape[1]):
            raise ValueError('D must have shape (outputs, inputs)')

        return Struct(A=A, B=B, C=C, D=D)

    def init(param) -> PyTree:
        dtype = jnp.result_type(param.A, param.B, param.C, param.D)
        return jnp.zeros(param.A.shape[0], dtype=dtype)

    def apply(param, state, input) -> tuple[PyTree, PyTree]:
        output = param.C @ state + param.D @ input
        state = param.A @ state + param.B @ input
        return state, output

    return Leaf(apply, param=param, init=init)
