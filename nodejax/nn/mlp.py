"""Multi-Layer Perceptron and Mixture-of-Experts blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import ambient
from nodejax.authoring import node_def, KeyStream
from nodejax.nn.linear import Linear
from nodejax.nn.activations import gelu


@ambient
def MLP(width: int, ratio: int):
    """The transformer's two-layer feed-forward: expand to ratio*width,
    gelu, project back to width. Plain composition of stock linears;
    width is the design decision, fan-ins derive from the offer."""
    return Linear(ratio * width) >> gelu >> Linear(width)


from nodejax.struct import Struct, Aux

@ambient
def MoE(hidden: int, experts: int):
    """Soft mixture-of-experts with an internal residual, written over a
    (B, hidden) batch: the load-balance statistic is a population
    quantity. Emits that statistic (experts * sum(mean_gate^2); 1.0 =
    uniform) and per-expert usage as AUX — the Aux convention; the
    loss decides what to do with it."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            router=0.2 * jax.random.normal(rng.next(), (hidden, experts)),
            w=0.5 * jax.random.normal(rng.next(), (experts, hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros((experts, hidden)))

    def apply(param: Struct, input: jax.Array) -> tuple[jax.Array, Aux]:
        gates = jax.nn.softmax(input @ param.router, axis=-1)              # (B, E)
        expert_out = jnp.tanh(jnp.einsum('bh,ehk->bek', input, param.w) + param.b)
        mixed = jnp.einsum('be,beh->bh', gates, expert_out)
        usage = jnp.mean(gates, axis=0)                                    # (E,)
        balance = experts * jnp.sum(usage ** 2)
        return input + mixed, Aux(balance=balance, usage=usage)

    return node_def(apply, param=param, name='moe')
