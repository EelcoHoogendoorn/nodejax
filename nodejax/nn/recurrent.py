"""Defines recurrent single-step cell factories (`RNN`, `GRU`, `MinGRU`, `LSTM`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def RNN(hidden: int) -> Node:
    """Advances an Elman `tanh` cell with carry width `hidden` and input width inferred from `node.input.shape[-1]`."""
    def param(node, rng) -> Struct:
        input_width = node.input.shape[-1]
        return Struct(
            wx=(0.5 * jax.random.normal(rng.next(), (input_width, hidden))
                / jnp.sqrt(input_width)),
            wh=(0.4 * jax.random.normal(rng.next(), (hidden, hidden))
                / jnp.sqrt(hidden)),
            b=jnp.zeros(hidden))

    def init(node, param) -> jax.Array:
        return jnp.zeros(
            node.input.shape[:-1] + (hidden,), dtype=param.b.dtype)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        hidden_value = jnp.tanh(input @ param.wx + state @ param.wh + param.b)
        return hidden_value, hidden_value

    return Leaf(apply, init=init, param=param)


@node
def GRU(hidden: int) -> Node:
    """Advances a gated recurrent unit with carry and output width `hidden`.

    Infers the input width from `node.input.shape[-1]` and computes update and reset gates from the concatenation of `input` and `state`.
    """
    def param(node, rng) -> Struct:
        input_width = node.input.shape[-1]
        fan_in = input_width + hidden

        def weight() -> jax.Array:
            return (0.4 * jax.random.normal(rng.next(), (fan_in, hidden))
                    / jnp.sqrt(fan_in))

        return Struct(
            update_weight=weight(),
            update_bias=jnp.zeros(hidden),
            reset_weight=weight(),
            reset_bias=jnp.zeros(hidden),
            candidate_weight=weight(),
            candidate_bias=jnp.zeros(hidden),
        )

    def init(node, param) -> jax.Array:
        return jnp.zeros(
            node.input.shape[:-1] + (hidden,),
            dtype=param.update_bias.dtype)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        gate_input = jnp.concatenate((input, state), axis=-1)
        update = jax.nn.sigmoid(
            gate_input @ param.update_weight + param.update_bias)
        reset = jax.nn.sigmoid(
            gate_input @ param.reset_weight + param.reset_bias)
        candidate_input = jnp.concatenate((input, reset * state), axis=-1)
        candidate = jnp.tanh(
            candidate_input @ param.candidate_weight + param.candidate_bias)
        hidden_value = (1.0 - update) * candidate + update * state
        return hidden_value, hidden_value

    return Leaf(apply, init=init, param=param)


@node
def MinGRU(hidden: int) -> Node:
    """Advances a minimal gated recurrent unit (Feng et al., 2024) with carry width `hidden`.

    Computes both the sigmoid update gate and the linear candidate from `input` alone (`node.input.shape[-1]` inferred), making the state transition `(1 - update) * state + update * candidate` diagonal and linear in `state`.
    """
    def param(node, rng) -> Struct:
        input_width = node.input.shape[-1]

        def weight() -> jax.Array:
            return (0.4 * jax.random.normal(rng.next(), (input_width, hidden))
                    / jnp.sqrt(input_width))

        return Struct(
            update_weight=weight(),
            update_bias=jnp.zeros(hidden),
            candidate_weight=weight(),
            candidate_bias=jnp.zeros(hidden),
        )

    def init(node, param) -> jax.Array:
        return jnp.zeros(
            node.input.shape[:-1] + (hidden,),
            dtype=param.update_bias.dtype)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        update = jax.nn.sigmoid(
            input @ param.update_weight + param.update_bias)
        candidate = input @ param.candidate_weight + param.candidate_bias
        hidden_value = (1.0 - update) * state + update * candidate
        return hidden_value, hidden_value

    return Leaf(apply, init=init, param=param)


@node
def LSTM(hidden: int) -> Node:
    """Advances a long short-term memory cell carrying `Struct(hidden=..., cell=...)` of width `hidden`.

    Projects `(input, state.hidden)` once to `4 * hidden` for the input, forget, candidate, and output gates, initializing the forget-gate bias to ones and the remaining gate biases to zeros.
    """
    def param(node, rng) -> Struct:
        input_width = node.input.shape[-1]
        fan_in = input_width + hidden
        weight = (0.4 * jax.random.normal(rng.next(), (fan_in, 4 * hidden))
                  / jnp.sqrt(fan_in))
        bias = jnp.concatenate((
            jnp.zeros(hidden),
            jnp.ones(hidden),
            jnp.zeros(hidden),
            jnp.zeros(hidden),
        ))
        return Struct(weight=weight, bias=bias)

    def init(node, param) -> Struct:
        zeros = jnp.zeros(
            node.input.shape[:-1] + (hidden,), dtype=param.bias.dtype)
        return Struct(hidden=zeros, cell=zeros)

    def apply(param, state, input) -> tuple[Struct, jax.Array]:
        projected = (jnp.concatenate((input, state.hidden), axis=-1)
                     @ param.weight + param.bias)
        input_gate, forget_gate, candidate, output_gate = jnp.split(
            projected, 4, axis=-1)
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)
        cell = forget_gate * state.cell + input_gate * candidate
        hidden_value = output_gate * jnp.tanh(cell)
        return Struct(hidden=hidden_value, cell=cell), hidden_value

    return Leaf(apply, init=init, param=param)
