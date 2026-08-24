"""A composition-oriented Flax NNX implementation of the tower.

The direct `tower_flax.py` specializes its scans, axes, carries, and MAML
orchestration to one program. This variant instead factors stacking, scanning,
and training into reusable operations, leaving `make_member` and `tower` to
show the complete model graph in one place.

The wrappers use two narrow local conventions. A step Module implements
``initial_carry()`` and ``(carry, input) -> (carry, output)``. A scanned Module
maps a complete input sequence to a complete output sequence. NNX preserves the
Modules and Variables passing through its graph-aware transforms; these
conventions supply the semantic facts that NNX itself does not standardize.

This is useful library design, not a failed attempt to imitate NodeJAX. Its
cost is visible: the file is roughly twice the size of the direct version. NNX
could also make MAML return another object satisfying an expanded local
protocol. Doing that generically would require the protocol to describe
cloning, parameter and state roles, optimizer state, inputs, auxiliary output,
axes, and randomness. That is possible; it is also progressively more of the
common contract NodeJAX provides once for every Node. This example leaves MAML
as a reusable training function to show where the local abstraction stops, not
where NNX reaches a capability limit.

This file uses the same model, task, and budget as ``tower_nodejax.py`` and
``tower_flax.py``.

Run directly: python -m nodejax.examples.comparisons.tower.tower_flax_reusable
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nodejax.examples.comparisons.tower.tower_common import (
    HIDDEN,
    INNER_LR,
    LAYERS,
    MEMBERS,
    META_STEPS,
    MOMENTUM,
    OUTER_LR,
    make_tasks,
)


# Reusable transform machinery

def stacked(make_module: Callable, n: int):
    """Construct ``n`` independently initialized Modules on axis zero."""
    @nnx.split_rngs(splits=n)
    @nnx.vmap(in_axes=0, out_axes=0)
    def make(rngs):
        return make_module(rngs)

    return make


def scan_layers(cells, signal, carries):
    """Feed a value through stacked recurrent cells."""
    @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def depth(cells, signal, carry):
        carry, output = cells(carry, signal)
        return output, carry

    return depth(cells, signal, carries)


def unroll(module, carry, inputs):
    """Run one step Module over the leading input axis."""
    module_axes = nnx.StateAxes({nnx.Param: None, ...: nnx.Carry})

    @nnx.scan(
        in_axes=(module_axes, nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
    )
    def over_time(module, carry, input):
        return module(carry, input)

    return over_time(module, carry, inputs)


def maml_fit(
    model,
    loss_fn: Callable,
    inner_lr: float,
    meta_opt,
    meta_steps: int,
    tasks: Any,
):
    """Adapt cloned Modules per task and train the shared initialization."""
    support_inputs, support_targets, query_inputs, query_targets = tasks
    adapt_axes = nnx.StateAxes({...: nnx.Carry})
    optimizer_axes = nnx.StateAxes({...: nnx.Carry})

    def per_task(
        initial,
        task_support_inputs,
        task_support_targets,
        query_input,
        query_target,
    ):
        adapted = nnx.clone(initial)
        inner_optimizer = nnx.Optimizer(
            adapted, optax.sgd(inner_lr), wrt=nnx.Param
        )

        @nnx.scan(
            in_axes=(adapt_axes, optimizer_axes, 0, 0),
            out_axes=0,
        )
        def adapt_step(adapted, optimizer, support_input, support_target):
            loss_value, grads = nnx.value_and_grad(
                lambda module: loss_fn(module(support_input), support_target)
            )(adapted)
            optimizer.update(adapted, grads)
            return loss_value

        adapt_step(
            adapted,
            inner_optimizer,
            task_support_inputs,
            task_support_targets,
        )
        return loss_fn(adapted(query_input), query_target)

    def meta_loss(current):
        losses = nnx.vmap(
            per_task,
            in_axes=(None, 0, 0, 0, 0),
        )(
            current,
            support_inputs,
            support_targets,
            query_inputs,
            query_targets,
        )
        return jnp.mean(losses)

    optimizer = nnx.Optimizer(model, meta_opt, wrt=nnx.Param)
    train_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def train_loop(current, optimizer, _):
        loss, grads = nnx.value_and_grad(meta_loss)(current)
        optimizer.update(current, grads)
        return loss

    return train_loop(model, optimizer, jnp.arange(meta_steps))


# Reusable structural wrappers

class Residual(nnx.Module):
    def __init__(self, body):
        self.body = body

    def initial_carry(self):
        return self.body.initial_carry()

    def __call__(self, carry, input):
        carry, output = self.body(carry, input)
        return carry, input + output


class Serial(nnx.Module):
    def __init__(self, *stages):
        self.stages = nnx.List(stages)

    def initial_carry(self):
        return tuple(stage.initial_carry() for stage in self.stages)

    def __call__(self, carry, input):
        next_carry = []
        for stage, stage_carry in zip(self.stages, carry):
            stage_carry, input = stage(stage_carry, input)
            next_carry.append(stage_carry)
        return tuple(next_carry), input


class DepthStack(nnx.Module):
    def __init__(self, layers):
        self.layers = layers

    def initial_carry(self):
        @nnx.vmap(in_axes=0, out_axes=0)
        def initialize(layer):
            return layer.initial_carry()

        return initialize(self.layers)

    def __call__(self, carry, input):
        output, carry = scan_layers(self.layers, input, carry)
        return carry, output


class Scanned(nnx.Module):
    def __init__(self, step):
        self.step = step

    def __call__(self, inputs):
        _, outputs = unroll(self.step, self.step.initial_carry(), inputs)
        return outputs


class Ensemble(nnx.Module):
    def __init__(self, members):
        self.members = members

    def __call__(self, inputs):
        return nnx.vmap(
            lambda member, sequence: member(sequence),
            in_axes=(0, None),
        )(self.members, inputs)


class Mean(nnx.Module):
    def __init__(self, body):
        self.body = body

    def __call__(self, inputs):
        return jnp.mean(self.body(inputs), axis=0)


# Leaf Modules

class Up(nnx.Module):
    def __init__(self, hidden: int, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.5 * jax.random.normal(rngs.params(), (hidden,))
        )

    def initial_carry(self):
        return ()

    def __call__(self, carry, input):
        return carry, self.weight[...] * input


class Norm(nnx.Module):
    def __init__(self, hidden: int, momentum: float):
        self.hidden = hidden
        self.momentum = momentum

    def initial_carry(self):
        return jnp.zeros(self.hidden), jnp.ones(self.hidden)

    def __call__(self, carry, input):
        mean, variance = carry
        output = (input - mean) / jnp.sqrt(variance + 1e-5)
        next_mean = (1 - self.momentum) * mean + self.momentum * input
        next_variance = (
            (1 - self.momentum) * variance
            + self.momentum * (input - mean) ** 2
        )
        return (next_mean, next_variance), output


class Cell(nnx.Module):
    def __init__(self, hidden: int, rngs: nnx.Rngs):
        key_x, key_h = jax.random.split(rngs.params())
        self.input_weight = nnx.Param(
            0.5 * jax.random.normal(key_x, (hidden,))
        )
        self.hidden_weight = nnx.Param(
            0.3
            * jax.random.normal(key_h, (hidden, hidden))
            / jnp.sqrt(hidden)
        )
        self.bias = nnx.Param(jnp.zeros(hidden))

    def initial_carry(self):
        return jnp.zeros(self.bias.shape)

    def __call__(self, carry, input):
        next_carry = jnp.tanh(
            self.input_weight[...] * input
            + self.hidden_weight[...] @ carry
            + self.bias[...]
        )
        return next_carry, next_carry


class Readout(nnx.Module):
    def __init__(self, hidden: int, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.1 * jax.random.normal(rngs.params(), (hidden,))
        )
        self.bias = nnx.Param(jnp.zeros(()))

    def initial_carry(self):
        return ()

    def __call__(self, carry, input):
        return carry, self.weight[...] @ input + self.bias[...]


# Concrete graph assembly

def make_member(rngs: nnx.Rngs):
    return Scanned(Serial(
        Up(HIDDEN, rngs),
        Norm(HIDDEN, MOMENTUM),
        DepthStack(
            stacked(
                lambda rngs: Residual(Cell(HIDDEN, rngs)),
                LAYERS,
            )(rngs)
        ),
        Readout(HIDDEN, rngs),
    ))


def tower(rngs: nnx.Rngs):
    members = stacked(make_member, MEMBERS)(rngs)
    return Mean(Ensemble(members))


def mse(output, target):
    return jnp.mean((output - target) ** 2)


def main():
    model = tower(nnx.Rngs(params=jax.random.PRNGKey(1)))
    losses = maml_fit(
        model,
        mse,
        INNER_LR,
        optax.adam(OUTER_LR),
        META_STEPS,
        make_tasks(jax.random.PRNGKey(2)),
    )
    print(f'tower loss {losses[0]:.4f} -> {losses[-1]:.4f}')
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
