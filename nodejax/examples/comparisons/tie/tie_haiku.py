"""Tied embeddings, the haiku side of the sharing comparison.

Self-contained: no nodejax import.

Haiku shares by PATH: params live in an ambient dict keyed by the module
name the call stack assigns, so two calls reach one parameter exactly
when they reach it under one path. The two views of the table are
therefore two methods of ONE module instance, and crossing the component
boundary means handing the instance across it: a separate Unembed module
would get a separate path and a separate table, with nothing warning
that the tie was lost. Where nodejax declares the tie structurally and
flax and torch hold it by object identity, haiku holds it by name, a
convention the params dict records and nothing checks.

Held as one path, the sharing cannot drift: the dict has one entry, so
copies=1 and drift=0 by construction.

Run directly:  python -m nodejax.examples.comparisons.tie.tie_haiku
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
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


class Table(hk.Module):
    """Both roles of one table. The path is the identity: each method
    reaches hk.get_parameter under this instance's name, so the params
    dict holds the table once however many roles read it."""

    def _table(self):
        return hk.get_parameter('table', (VOCAB, DIM),
                                init=hk.initializers.RandomNormal(0.3))

    def embed(self, token):
        return self._table()[token]

    def unembed(self, h):
        return h @ self._table().T


class RNNCell(hk.Module):
    def __call__(self, h, x):
        wh = hk.get_parameter('wh', (DIM, DIM),
                              init=hk.initializers.RandomNormal(0.5 / np.sqrt(DIM)))
        return jnp.tanh(x + wh @ h)


def rollout(ids: jax.Array):
    # the sharing: ONE Table instance handed to both ends of the model.
    # Two instances would be two paths and two tables, silently
    table = Table()
    cell = RNNCell()

    def step(h, token):
        h2 = cell(h, table.embed(token))
        return h2, table.unembed(h2)

    _, logits = hk.scan(step, jnp.zeros(DIM), ids)
    return logits


def main() -> None:
    prev, cur = make_data(np.random.RandomState(0))
    f = hk.without_apply_rng(hk.transform(rollout))
    params = f.init(jax.random.PRNGKey(0), prev)
    opt = optax.adam(LR)

    tables = sum(leaf.shape == (VOCAB, DIM) for leaf in jax.tree.leaves(params))

    def loss_of(params):
        return xent(f.apply(params, prev), cur)

    @jax.jit
    def train(params, opt_state):
        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_of)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss
        return jax.lax.scan(step, (params, opt_state), None, length=STEPS)

    (params, _), losses = train(params, opt.init(params))
    print(f'{"haiku":14s} table_copies={tables} drift=0.00e+00 '
          f'loss {float(losses[0]):.2f} -> {float(losses[-1]):.2f}', flush=True)


if __name__ == '__main__':
    main()
