"""Tied embeddings, the equinox side of the sharing comparison.

Self-contained: no nodejax import.

The model is built from components (an Embed block, an RNN cell, an
Unembed head), so the tie must cross a composition boundary: the two
views of the table live in different modules of the tree.

Equinox modules ARE pytrees, and this file measures the cost of that
choice for sharing: an aliased array DUPLICATES on flatten. The naive
spelling below hands the same table to both component constructors; the
optimizer sees two leaves, updates them from their separate gradients,
and the views DRIFT apart from the first step. This is the exact
failure the flax nnx documentation names as its reason modules must not
be pytrees.

eqx.nn.Shared is equinox's repair: it removes the aliased node at
construction and reinserts it at call time. The where/get lenses reach
across the component boundary by path, which is the module-shaped form
of what nodejax's tie declares by member name.

Run directly:  python -m nodejax.examples.comparisons.tie.tie_equinox
"""

import numpy as np
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

VOCAB, DIM = 8, 6
POSITIONS, STEPS = 256, 200
LR = 0.03
CONCENTRATION = 2.0


def make_data(rs):
    logits = CONCENTRATION * rs.standard_normal((VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros(POSITIONS + 1, dtype=np.int64)
    state = rs.randint(VOCAB)
    for t in range(POSITIONS + 1):
        tokens[t] = state
        state = (P[state].cumsum() > rs.random()).argmax()
    return jnp.asarray(tokens[:-1], jnp.int32), jnp.asarray(tokens[1:], jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Embed(eqx.Module):
    table: jax.Array

    def __call__(self, token):
        return self.table[token]


class RNNCell(eqx.Module):
    wh: jax.Array

    def __call__(self, h, x):
        h = jnp.tanh(x + self.wh @ h)
        return h, h


class Unembed(eqx.Module):
    table: jax.Array

    def __call__(self, h):
        return h @ self.table.T


class LM(eqx.Module):
    embed: Embed
    cell: RNNCell
    unembed: Unembed

    def rollout(self, ids):
        def step(h, token):
            h, y = self.cell(h, self.embed(token))
            return h, self.unembed(y)

        _, logits = jax.lax.scan(step, jnp.zeros(DIM), ids)
        return logits


def fit(model, prev: jax.Array, cur: jax.Array, forward: Callable):
    """The filtering ceremony: the model splits into arrays and statics so
    the training scan can carry it (a SharedNode sentinel is not an array),
    and recombines inside the loss. Every transform boundary repeats it."""
    opt = optax.adam(LR)
    arrays, static = eqx.partition(model, eqx.is_array)

    @jax.jit
    def train(arrays, opt_state):
        def step(carry, _):
            arrays, opt_state = carry

            def loss_of(arrays):
                merged = eqx.combine(arrays, static)
                return xent(forward(merged).rollout(prev), cur)

            loss, grads = jax.value_and_grad(loss_of)(arrays)
            updates, opt_state = opt.update(grads, opt_state, arrays)
            return (eqx.apply_updates(arrays, updates), opt_state), loss
        return jax.lax.scan(step, (arrays, opt_state), None, length=STEPS)

    (arrays, _), losses = train(arrays, opt.init(arrays))
    return eqx.combine(arrays, static), losses


def line(name: str, tables, drift, losses):
    print(f'{name:14s} table_copies={tables} drift={drift:.2e} '
          f'loss {float(losses[0]):.2f} -> {float(losses[-1]):.2f}', flush=True)


def main() -> None:
    prev, cur = make_data(np.random.RandomState(0))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    table = 0.3 * jax.random.normal(k1, (VOCAB, DIM))
    cell = RNNCell(wh=0.5 * jax.random.normal(k2, (DIM, DIM)) / jnp.sqrt(DIM))

    # THE NAIVE SPELLING: one array handed to both component constructors.
    # Flatten duplicates it, the optimizer updates two tables, and the
    # alias silently breaks
    naive = LM(embed=Embed(table), cell=cell, unembed=Unembed(table))
    tables = sum(leaf.shape == (VOCAB, DIM) for leaf in jax.tree.leaves(naive))
    fitted, losses = fit(naive, prev, cur, forward=lambda m: m)
    drift = float(jnp.max(jnp.abs(fitted.embed.table - fitted.unembed.table)))
    # note the LOWER loss: two untied tables fit better than one, which is
    # exactly why the break is silent. Nothing crashes; the model just
    # stops being the model that was declared
    line('equinox naive', tables, drift, losses)

    # THE REPAIR: eqx.nn.Shared removes the aliased node at construction
    # and reinserts it at call time, the lenses reaching across the
    # component boundary by path
    shared = eqx.nn.Shared(naive, where=lambda m: m.unembed.table,
                           get=lambda m: m.embed.table)
    tables = sum(eqx.is_array(leaf) and leaf.shape == (VOCAB, DIM)
                 for leaf in jax.tree.leaves(shared))
    fitted, losses = fit(shared, prev, cur, forward=lambda s: s())
    inner = fitted()
    drift = float(jnp.max(jnp.abs(inner.embed.table - inner.unembed.table)))
    line('equinox Shared', tables, drift, losses)


if __name__ == '__main__':
    main()
