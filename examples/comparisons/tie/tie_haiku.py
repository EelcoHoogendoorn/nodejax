"""Tied input/output embeddings in Haiku.

Haiku keys parameters by module path inside the ambient `hk.transform` dictionary. Sharing one table across input and output views requires putting both `embed` and `unembed` methods on a single `Table(hk.Module)` instance passed to both call sites; instantiating two separate modules silently creates two distinct parameter entries.

Run directly with `python -m examples.comparisons.tie.tie_haiku`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from examples.comparisons.tie.tie_common import (
    DIM, LR, STEPS, VOCAB, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Table(hk.Module):
    def _table(self) -> jax.Array:
        return hk.get_parameter(
            'table', (VOCAB, DIM), init=hk.initializers.RandomNormal(0.3)
        )

    def embed(self, token: jax.Array) -> jax.Array:
        return self._table()[token]

    def unembed(self, h: jax.Array) -> jax.Array:
        return h @ self._table().T


class RNNCell(hk.Module):
    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        wh = hk.get_parameter(
            'wh', (DIM, DIM), init=hk.initializers.RandomNormal(0.5 / float(np.sqrt(DIM)))
        )
        return jnp.tanh(x + wh @ h)


def rollout(ids: jax.Array) -> jax.Array:
    table = Table()
    cell = RNNCell()

    def step(h, token):
        h2 = cell(h, table.embed(token))
        return h2, table.unembed(h2)

    _, logits = hk.scan(step, jnp.zeros(DIM), ids)
    return logits


def main() -> None:
    prev_np, cur_np = make_data(np.random.RandomState(0))
    prev, cur = jnp.asarray(prev_np), jnp.asarray(cur_np)
    f = hk.without_apply_rng(hk.transform(rollout))
    params = f.init(jax.random.PRNGKey(0), prev)
    opt = optax.adam(LR)

    tables = sum(leaf.shape == (VOCAB, DIM) for leaf in jax.tree.leaves(params))

    @jax.jit
    def train(p0, opt_state):
        def step(carry, _):
            cur_p, cur_opt = carry
            loss, grads = jax.value_and_grad(lambda p: xent(f.apply(p, prev), cur))(cur_p)
            updates, next_opt = opt.update(grads, cur_opt, cur_p)
            return (optax.apply_updates(cur_p, updates), next_opt), loss

        return jax.lax.scan(step, (p0, opt_state), None, length=STEPS)

    (params, _), losses = train(params, opt.init(params))
    report('haiku', tables, 0.0, float(losses[0]), float(losses[-1]))


if __name__ == '__main__':
    main()
