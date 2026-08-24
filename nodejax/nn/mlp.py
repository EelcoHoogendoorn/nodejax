"""Multi-Layer Perceptron and Mixture-of-Experts blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.node import Node
from nodejax.struct import Struct
from nodejax.binding import (Aux)
from nodejax.ambient import node
from nodejax.authoring import Leaf
from nodejax.nn.linear import Linear
from nodejax.nn.activations import gelu


@node
def MLP(width: int, ratio: int):
    """The transformer's two-layer feed-forward: expand to ratio*width,
    gelu, project back to width. Plain composition of stock linears;
    width is the design decision, fan-ins derive from the resolved input spec."""
    return Linear(ratio * width) >> gelu >> Linear(width)


@node
def SwiGLU(width: int, ratio: int) -> Node:
    """SwiGLU feed-forward block with an inferred input width.

    Two projections expand to `ratio * width`. One passes through SiLU and
    gates the other elementwise, then a third projection returns `width`
    features.
    """
    def param(node, rng) -> Struct:
        input_width = node.input.shape[-1]
        hidden = ratio * width
        return Struct(
            gate_weight=(jax.random.normal(rng.next(), (input_width, hidden))
                         / jnp.sqrt(input_width)),
            gate_bias=jnp.zeros(hidden),
            value_weight=(jax.random.normal(rng.next(), (input_width, hidden))
                          / jnp.sqrt(input_width)),
            value_bias=jnp.zeros(hidden),
            output_weight=(jax.random.normal(rng.next(), (hidden, width))
                           / jnp.sqrt(hidden)),
            output_bias=jnp.zeros(width),
        )

    def apply(param, input) -> jax.Array:
        gate = jax.nn.silu(input @ param.gate_weight + param.gate_bias)
        value = input @ param.value_weight + param.value_bias
        return (gate * value) @ param.output_weight + param.output_bias

    return Leaf(apply, param=param)


@node(name='moe')
def MoE(hidden: int, experts: int):
    """Soft mixture-of-experts with an internal residual, written over a
    (B, hidden) batch: the load-balance statistic is a population
    quantity. Emits that statistic (experts * sum(mean_gate^2); 1.0 =
    uniform) and per-expert usage as AUX — the Aux convention; the
    loss decides what to do with it."""
    def param(rng) -> Struct:
        return Struct(
            router=0.2 * jax.random.normal(rng.next(), (hidden, experts)),
            w=0.5 * jax.random.normal(rng.next(), (experts, hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros((experts, hidden)))

    def apply(param, input) -> tuple[jax.Array, Aux]:
        gates = jax.nn.softmax(input @ param.router, axis=-1)              # (B, E)
        expert_out = jnp.tanh(jnp.einsum('bh,ehk->bek', input, param.w) + param.b)
        mixed = jnp.einsum('be,beh->bh', gates, expert_out)
        usage = jnp.mean(gates, axis=0)                                    # (E,)
        balance = experts * jnp.sum(usage ** 2)
        return input + mixed, Aux(balance=balance, usage=usage)

    return Leaf(apply, param=param)
