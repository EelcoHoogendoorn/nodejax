"""Tied input/output embeddings in Flax NNX.

Passing the same `nnx.Param` instance to `Embed` and `Unembed` shares the table by Python object identity inside NNX's reference-aware graph. `nnx.state` and `nnx.Optimizer` deduplicate the shared reference so the optimizer updates a single array leaf with zero drift.

Run directly with `python -m examples.comparisons.tie.tie_nnx`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.comparisons.tie.tie_common import (
    DIM, LR, STEPS, VOCAB, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Embed(nnx.Module):
    def __init__(self, table: nnx.Param):
        self.table = table

    def __call__(self, token: jax.Array) -> jax.Array:
        return self.table[token]


class RNNCell(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.wh = nnx.Param(0.5 * jax.random.normal(rngs.params(), (DIM, DIM)) / jnp.sqrt(DIM))

    def __call__(self, h: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(x + self.wh[...] @ h)
        return h, h


class Unembed(nnx.Module):
    def __init__(self, table: nnx.Param):
        self.table = table

    def __call__(self, h: jax.Array) -> jax.Array:
        return h @ self.table[...].T


class LM(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        table = nnx.Param(0.3 * jax.random.normal(rngs.params(), (VOCAB, DIM)))
        self.embed = Embed(table)
        self.cell = RNNCell(rngs)
        self.unembed = Unembed(table)

    def rollout(self, ids: jax.Array) -> jax.Array:
        @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def over_time(lm: LM, h: jax.Array, token: jax.Array):
            h, y = lm.cell(h, lm.embed(token))
            return h, lm.unembed(y)

        _, logits = over_time(self, jnp.zeros(DIM), ids)
        return logits


def main() -> None:
    prev_np, cur_np = make_data(np.random.RandomState(0))
    prev, cur = jnp.asarray(prev_np), jnp.asarray(cur_np)
    model = LM(nnx.Rngs(params=jax.random.PRNGKey(0)))
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    tables = [leaf for leaf in jax.tree.leaves(nnx.state(model)) if leaf.shape == (VOCAB, DIM)]

    @nnx.scan(
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.StateAxes({...: nnx.Carry}), 0),
        out_axes=0,
    )
    def train_loop(m: LM, opt: nnx.Optimizer, _):
        loss, grads = nnx.value_and_grad(lambda mod: xent(mod.rollout(prev), cur))(m)
        opt.update(m, grads)
        return loss

    losses = train_loop(model, optimizer, jnp.arange(STEPS))
    drift = float(jnp.max(jnp.abs(model.embed.table[...] - model.unembed.table[...])))
    report('nnx', len(tables), drift, float(losses[0]), float(losses[-1]))


if __name__ == '__main__':
    main()
