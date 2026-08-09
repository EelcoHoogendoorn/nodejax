"""Elementwise activation function blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.authoring import node_def


gelu = node_def(lambda input: jax.nn.gelu(input), name='gelu')
relu = node_def(lambda input: jax.nn.relu(input), name='relu')
silu = node_def(lambda input: jax.nn.silu(input), name='silu')
sigmoid = node_def(lambda input: jax.nn.sigmoid(input), name='sigmoid')
tanh = node_def(lambda input: jnp.tanh(input), name='tanh')
