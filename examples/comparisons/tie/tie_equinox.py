"""Tied input/output embeddings in Equinox: PyTree aliasing failure and `eqx.nn.Shared`.

Because Equinox modules are plain PyTrees, passing the same `jax.Array` to `Embed` and `Unembed` duplicates the leaf on `jax.tree.leaves`, causing the optimizer to update two independent copies that drift apart immediately. Wrapping the model in `eqx.nn.Shared` replaces the target leaf with a sentinel and restores single-table sharing.

Run directly with `python -m examples.comparisons.tie.tie_equinox`.
"""

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from examples.comparisons.tie.tie_common import (
    DIM, LR, STEPS, VOCAB, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Embed(eqx.Module):
    table: jax.Array

    def __call__(self, token: jax.Array) -> jax.Array:
        return self.table[token]


class RNNCell(eqx.Module):
    wh: jax.Array

    def __call__(self, h: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(x + self.wh @ h)
        return h, h


class Unembed(eqx.Module):
    table: jax.Array

    def __call__(self, h: jax.Array) -> jax.Array:
        return h @ self.table.T


class LM(eqx.Module):
    embed: Embed
    cell: RNNCell
    unembed: Unembed

    def rollout(self, ids: jax.Array) -> jax.Array:
        def step(h, token):
            h, y = self.cell(h, self.embed(token))
            return h, self.unembed(y)

        _, logits = jax.lax.scan(step, jnp.zeros(DIM), ids)
        return logits


def fit(model, prev: jax.Array, cur: jax.Array, forward: Callable):
    opt = optax.adam(LR)
    arrays, static = eqx.partition(model, eqx.is_array)

    @jax.jit
    def train(arrs, opt_state):
        def step(carry, _):
            cur_arrs, cur_opt = carry
            loss, grads = jax.value_and_grad(
                lambda a: xent(forward(eqx.combine(a, static)).rollout(prev), cur)
            )(cur_arrs)
            updates, next_opt = opt.update(grads, cur_opt, cur_arrs)
            return (eqx.apply_updates(cur_arrs, updates), next_opt), loss

        return jax.lax.scan(step, (arrs, opt_state), None, length=STEPS)

    (arrays, _), losses = train(arrays, opt.init(arrays))
    return eqx.combine(arrays, static), losses


def main() -> None:
    prev_np, cur_np = make_data(np.random.RandomState(0))
    prev, cur = jnp.asarray(prev_np), jnp.asarray(cur_np)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    table = 0.3 * jax.random.normal(k1, (VOCAB, DIM))
    cell = RNNCell(wh=0.5 * jax.random.normal(k2, (DIM, DIM)) / jnp.sqrt(DIM))

    naive = LM(embed=Embed(table), cell=cell, unembed=Unembed(table))
    naive_copies = sum(leaf.shape == (VOCAB, DIM) for leaf in jax.tree.leaves(naive))
    fitted_naive, losses_naive = fit(naive, prev, cur, forward=lambda m: m)
    drift_naive = float(jnp.max(jnp.abs(fitted_naive.embed.table - fitted_naive.unembed.table)))
    report(
        'equinox naive', naive_copies, drift_naive,
        float(losses_naive[0]), float(losses_naive[-1]), expect_tied=False,
    )

    shared = eqx.nn.Shared(
        naive, where=lambda m: m.unembed.table, get=lambda m: m.embed.table
    )
    shared_copies = sum(
        eqx.is_array(leaf) and leaf.shape == (VOCAB, DIM)
        for leaf in jax.tree.leaves(shared)
    )
    fitted_shared, losses_shared = fit(shared, prev, cur, forward=lambda s: s())
    inner = fitted_shared()
    drift_shared = float(jnp.max(jnp.abs(inner.embed.table - inner.unembed.table)))
    report(
        'equinox Shared', shared_copies, drift_shared,
        float(losses_shared[0]), float(losses_shared[-1]), expect_tied=True,
    )


if __name__ == '__main__':
    main()
