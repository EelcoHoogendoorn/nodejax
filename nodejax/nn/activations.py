"""Elementwise activation function blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.authoring import Leaf


gelu = Leaf(lambda input: jax.nn.gelu(input), name='gelu')
relu = Leaf(lambda input: jax.nn.relu(input), name='relu')
silu = Leaf(lambda input: jax.nn.silu(input), name='silu')
sigmoid = Leaf(lambda input: jax.nn.sigmoid(input), name='sigmoid')
tanh = Leaf(lambda input: jnp.tanh(input), name='tanh')
elu = Leaf(lambda input: jax.nn.elu(input), name='elu')
leaky_relu = Leaf(lambda input: jax.nn.leaky_relu(input), name='leaky_relu')
softplus = Leaf(lambda input: jax.nn.softplus(input), name='softplus')
softmax = Leaf(lambda input: jax.nn.softmax(input, axis=-1), name='softmax')
log_softmax = Leaf(
    lambda input: jax.nn.log_softmax(input, axis=-1), name='log_softmax')
identity = Leaf(lambda input: input, name='identity')
