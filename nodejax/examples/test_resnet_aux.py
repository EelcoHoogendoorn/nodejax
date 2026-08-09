"""Stacked ResNet example with per-layer auxiliary activation losses.

Demonstrates:
1. Every layer in a residual block sows its L2 activation penalty via `ndef.sow(act_l2=...)`.
2. `residual(...)` preserves skip connections over the main data signal while passing `aux` through.
3. `stack(...)` scans the residual block over depth=L, automatically stacking the sown `act_l2` into shape `(depth,)`.
4. `train_step` trains the stacked ResNet with total loss = MSE(pred, target) + lambda * sum(aux.act_l2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import node_def, residual, stack, train_step, split_aux, Aux
from nodejax.nn import linear
from nodejax.util import mse, tile


def SownBlock(dim: int):
    """A linear block that emits its L2 activation penalty as Aux."""
    def apply(param, input):
        y = jax.nn.gelu(input @ param.w + param.b)
        return y, Aux(act_l2=jnp.mean(y ** 2))

    def param(rng):
        return Struct(w=jax.random.normal(rng.next(), (dim, dim)) / jnp.sqrt(dim),
                      b=jnp.zeros(dim))

    return node_def(apply, param=param, name='sown_block')


def StackedResNet(dim: int, depth: int):
    """A stacked ResNet: residual(SownBlock) stacked to depth L."""
    return stack(residual(SownBlock(dim)), depth)


def test_stacked_resnet_aux_accumulation():
    """Verify that stacked ResNet emits main output cleanly while aux carries stacked (depth,) L2 penalties."""
    depth = 4
    dim = 8
    model_def = StackedResNet(dim=dim, depth=depth)
    model = model_def.parameterize(rng=jax.random.PRNGKey(0))

    x = jnp.ones(dim)
    out, aux = model.apply(x)

    # Output shape matches input dimension
    assert out.shape == (dim,)
    # Aux carries per-layer L2 penalties stacked over depth
    assert aux.act_l2.shape == (depth,)


def test_stacked_resnet_training_with_aux_loss():
    """Train stacked ResNet where total loss includes per-layer sown activation penalties."""
    depth = 3
    dim = 4
    model_def = StackedResNet(dim=dim, depth=depth)

    def loss_fn(output, target):
        pred, aux = output
        total_act_penalty = jnp.sum(aux.act_l2)
        return mse(pred, target) + 0.01 * total_act_penalty

    bound_model = model_def.parameterize(rng=jax.random.PRNGKey(42))
    trainer = train_step(model_def, loss_fn, optax.adam(1e-2))
    state = trainer.init(model=bound_model.param)

    steps = 100
    x_data = tile(jnp.ones(dim), steps)
    y_target = tile(jnp.ones(dim) * 2.0, steps)

    final_state, losses = trainer.scan(state, Struct(input=x_data, target=y_target))

    # Verification: loss decreased
    assert losses[-1] < losses[0]
