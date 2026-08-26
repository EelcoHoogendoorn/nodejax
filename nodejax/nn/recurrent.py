"""Recurrent neural network cell blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.node import Node
from nodejax.struct import Struct
from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf


@node
def RNN(hidden: int) -> Node:
    """Elman cell with a hidden-width carry.

    The input width comes from the resolved input spec. The hidden width is a
    design choice and may differ from it.
    """
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
    """Gated recurrent unit with an inferred input width.

    The update and reset gates see both the current input and previous hidden
    value. State and output both have the requested hidden width.
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
    """Minimal gated recurrent unit (Feng et al. 2024) with an inferred
    input width.

    Both the update gate and the candidate read only the current input,
    never the state, so the state transition is linear and diagonal with
    decay factors sigmoid-bounded inside (0, 1): transport along time is
    contractive at any horizon. Curvature must come from blocks placed
    around the cell.
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
    """Long short-term memory cell with hidden and cell state.

    The four gates share one affine projection. The forget-gate bias starts at
    one, while the other gate biases start at zero.
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
