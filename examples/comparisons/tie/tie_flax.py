"""Tied embeddings, the flax nnx side of the sharing comparison.

Self-contained: no nodejax import.

The model is built from components (an Embed block, an RNN cell, an
Unembed head), so the tie must cross a composition boundary: the two
views of the table live in different modules.

NNX shares by object reference. Hand the same Variable to two component
constructors and the graph records one object with two uses. Its graph-aware
transforms and optimizer preserve that reference, and `nnx.state` exposes one
table. This file checks the resulting behavior: one optimizer-visible table
and no drift between the two uses.

The tradeoff is representational. The sharing relationship belongs to NNX's
reference-aware graph rather than to the shape of a plain JAX value tree.
NodeJAX records the relationship as a structural parameter transformation;
both mechanisms keep one trainable value, but they place that fact in different
representations.

Run directly:  python -m examples.comparisons.tie.tie_flax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

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


class Embed(nnx.Module):
    def __init__(self, table: nnx.Param):
        self.table = table

    def __call__(self, token):
        return self.table[token]


class RNNCell(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.wh = nnx.Param(0.5 * jax.random.normal(rngs.params(), (DIM, DIM)) / jnp.sqrt(DIM))

    def __call__(self, h, x):
        h = jnp.tanh(x + self.wh[...] @ h)
        return h, h


class Unembed(nnx.Module):
    def __init__(self, table: nnx.Param):
        self.table = table

    def __call__(self, h):
        return h @ self.table[...].T


class LM(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        table = nnx.Param(0.3 * jax.random.normal(rngs.params(), (VOCAB, DIM)))
        # the sharing: the SAME Variable object handed to two component
        # constructors. The graph records the aliasing; the tree never
        # sees it
        self.embed = Embed(table)
        self.cell = RNNCell(rngs)
        self.unembed = Unembed(table)

    def rollout(self, ids):
        @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def over_time(lm, h, token):
            h, y = lm.cell(h, lm.embed(token))
            return h, lm.unembed(y)

        _, logits = over_time(self, jnp.zeros(DIM), ids)
        return logits


def main() -> None:
    prev, cur = make_data(np.random.RandomState(0))
    model = LM(nnx.Rngs(params=jax.random.PRNGKey(0)))
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    tables = [leaf for leaf in jax.tree.leaves(nnx.state(model))
              if leaf.shape == (VOCAB, DIM)]

    @nnx.scan(in_axes=(nnx.StateAxes({...: nnx.Carry}),
                       nnx.StateAxes({...: nnx.Carry}), 0), out_axes=0)
    def train_loop(model, optimizer, _):
        loss, grads = nnx.value_and_grad(lambda m: xent(m.rollout(prev), cur))(model)
        optimizer.update(model, grads)
        return loss

    losses = train_loop(model, optimizer, jnp.arange(STEPS))
    drift = float(jnp.max(jnp.abs(model.embed.table[...] - model.unembed.table[...])))
    print(f'{"flax nnx":14s} table_copies={len(tables)} drift={drift:.2e} '
          f'loss {float(losses[0]):.2f} -> {float(losses[-1]):.2f}', flush=True)


if __name__ == '__main__':
    main()
