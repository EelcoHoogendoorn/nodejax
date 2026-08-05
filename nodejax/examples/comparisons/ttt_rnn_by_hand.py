"""meta_comparison's ttt-rnn row, hand-rolled in raw JAX.

Self-contained: no nodejax import. The same model — a tanh rnn whose
weights are fast state, updated by one self-supervised gradient step
per sample (predict-then-update), initialization and per-weight rates
meta-learned — on the same two-sine task family, same budget, same
scoring. Exists as the comparison exhibit for the pitch: what the
library row expresses as

    model = batch(scan(ttt(rnn_def(HIDDEN), mse, 0.01)))

is here the whole file. Everything the transforms carry implicitly is
explicit below: the (weights, hidden) carry tuple, the rate tree
threaded by closure, the per-task vmap, the meta-training scan, the
init/apply split. It is worth noting what the raw form does NOT
make hard: one fixed configuration, hand-rolled once, is manageable.
The tax is structural: every row-swap of meta_comparison (linear,
mlp, ssm, a stacked pipe as the inner) is here a rewrite, and none of
batch/ensemble/persist/stack exists to be composed.

Run directly:  python -m nodejax.examples.ttt_rnn_by_hand
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

LAGS = 8
STREAM, SUPPORT = 192, 128
HIDDEN = 16
TASKS, META_STEPS = 8, 400
TTT_LR0, META_LR = 0.01, 1e-3
NOISE = 0.05
QUERY0 = SUPPORT - LAGS


def make_tasks(rs, n_tasks):
    t = np.arange(STREAM)[None, :]
    def draw(lo, hi):
        return rs.uniform(lo, hi, (n_tasks, 1))
    x = (draw(0.5, 1.5) * np.sin(2 * np.pi * draw(0.02, 0.1) * t + draw(0, 2 * np.pi))
         + draw(0.2, 0.8) * np.sin(2 * np.pi * draw(0.1, 0.25) * t + draw(0, 2 * np.pi))
         + NOISE * rs.standard_normal((n_tasks, STREAM))).astype(np.float32)
    M = STREAM - LAGS
    lags = np.stack([x[:, i:i + LAGS] for i in range(M)], axis=1)
    return dict(input=jnp.asarray(lags), target=jnp.asarray(x[:, LAGS:]))


def rnn_init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(win=0.5 * jax.random.normal(k1, (LAGS, HIDDEN)),
                wh=0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN),
                b=jnp.zeros(HIDDEN),
                wout=0.3 * jax.random.normal(k3, (HIDDEN,)))


def rnn_apply(w, h, x):
    h = jnp.tanh(x @ w['win'] + w['wh'] @ h + w['b'])
    return h, w['wout'] @ h


def forecast_stream(theta, stream):
    """One task: scan the fast-weight cell down the sample stream."""
    def cell(carry, sample):
        w, h = carry
        def selfsup(wf):
            h2, pred = rnn_apply(wf, h, sample['input'])
            return (pred - sample['target']) ** 2, (h2, pred)
        grads, (h2, pred) = jax.grad(selfsup, has_aux=True)(w)
        w2 = jax.tree.map(lambda wi, g, lr: wi - lr * g, w, grads, theta['lr'])
        return (w2, h2), pred

    (_, _), preds = jax.lax.scan(cell, (theta['init'], jnp.zeros(HIDDEN)), stream)
    return preds


def query_mse(theta, tasks):
    preds = jax.vmap(lambda s: forecast_stream(theta, s))(tasks)
    return jnp.mean((preds[:, QUERY0:] - tasks['target'][:, QUERY0:]) ** 2)


def main():
    init = rnn_init(jax.random.PRNGKey(0))
    theta = dict(init=init, lr=jax.tree.map(lambda x: jnp.full_like(x, TTT_LR0), init))
    opt = optax.adam(META_LR)

    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    stream = jax.tree.map(lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:]), train)

    def meta_step(carry, batch):
        theta, opt_state = carry
        loss, grads = jax.value_and_grad(query_mse)(theta, batch)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return (optax.apply_updates(theta, updates), opt_state), loss

    (theta, _), losses = jax.lax.scan(meta_step, (theta, opt.init(model=theta)), stream)

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    n = sum(x.size for x in jax.tree.leaves(theta))
    print(f'ttt-rnn by hand: weights={n} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query mse {query_mse(theta, tasks):.4f}')


if __name__ == '__main__':
    main()
