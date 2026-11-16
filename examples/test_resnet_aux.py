"""Stacked residual network with per-layer `Aux` activation penalties.

Each `SownBlock` emits its mean squared activation in `Aux(act_l2=...)`.
`residual` adds skip connections around the primary output while forwarding
`Aux` unchanged, `stack` scans the residual block across `depth` layers so
`aux.act_l2` acquires shape `(depth,)`, and `train_step` passes the accumulated
`Aux` record into a loss function declaring keyword `aux`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import Node, node, Leaf, residual, stack, train_step, trained, Aux, tile
from examples.util import mse


@node
def SownBlock(dim: int) -> Node:
    """Applies a GELU linear transformation of width `dim` and emits mean squared activation in `Aux`."""
    def apply(param, input):
        y = jax.nn.gelu(input @ param.w + param.b)
        return y, Aux(act_l2=jnp.mean(y ** 2))

    def param(rng):
        return Struct(w=jax.random.normal(rng.next(), (dim, dim)) / jnp.sqrt(dim),
                      b=jnp.zeros(dim))

    return Leaf(apply, param=param)


@node
def StackedResNet(dim: int, depth: int) -> Node:
    """Stacks `depth` residual `SownBlock` layers of width `dim`."""
    return stack(residual(SownBlock(dim)), depth)


def test_stacked_resnet_aux_accumulation() -> None:
    """Verifies that `StackedResNet` preserves feature width `dim` and stacks `act_l2` penalties to shape `(depth,)`."""
    depth = 4
    dim = 8
    model_def = StackedResNet(dim=dim, depth=depth)
    model = model_def.parameterize(rng=jax.random.PRNGKey(0))

    x = jnp.ones(dim)
    out, aux = model.apply(x)

    assert out.shape == (dim,)
    assert aux.act_l2.shape == (depth,)


def test_stacked_resnet_training_with_aux_loss() -> None:
    """Verifies that `train_step` updates parameters under a loss reading summed per-layer `Aux` penalties."""
    depth = 3
    dim = 4
    model_def = StackedResNet(dim=dim, depth=depth)

    def loss_fn(output, target, *, aux):
        total_act_penalty = jnp.sum(aux.act_l2)
        return mse(output, target) + 0.01 * total_act_penalty

    initial = model_def.parameterize(rng=jax.random.PRNGKey(42)).initialize()
    trainer = train_step(initial, loss_fn, optax.adam(1e-2))
    steps = 3
    final, aux = trained(trainer).apply(input=tile(jnp.ones(dim), steps),
                                        target=tile(jnp.ones(dim) * 2.0, steps))
    assert aux.loss.shape == (steps,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, initial.param))
