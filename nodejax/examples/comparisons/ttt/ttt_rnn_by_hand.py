"""ttt_nodejax's ttt-rnn row, hand-rolled in raw JAX.

Self-contained: no nodejax import. The same model, a tanh rnn whose
weights are fast state, updated by one next-token gradient step per
token (predict-then-update), initialization and per-weight rates
meta-learned, on the same Markov task family, same budget, same
scoring. Exists as the comparison exhibit for the pitch: what the
library row expresses as

    model = batch(scanned(next_step_ttt(train_step(
        RNN(VOCAB, HIDDEN), xent, learned_sgd(TTT_LR0)))))

is here the whole file. Everything the transforms carry implicitly is
explicit below: the (weights, hidden, previous token) carry triple,
where the third slot is the next_step register spelled by hand; the
rate tree threaded by closure; the per-task vmap; the meta-training
scan; the init/apply split. It is worth noting what the raw form does
NOT make hard: one fixed configuration, hand-rolled once, is
manageable. The tax is structural: every row-swap of ttt_nodejax (a
different inner model, a different pairing, a stacked pipe as the
inner) is here a rewrite, and none of batch/scanned/stack exists to be
composed.

Run directly:  python -m nodejax.examples.comparisons.ttt.ttt_rnn_by_hand
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

VOCAB, HIDDEN = 8, 16
STREAM, SUPPORT = 192, 128
TASKS, META_STEPS = 8, 300
TTT_LR0, META_LR = 0.05, 1e-3
CONCENTRATION = 2.0
QUERY0 = SUPPORT


def make_tasks(rs, n_tasks: int):
    logits = CONCENTRATION * rs.standard_normal((n_tasks, VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros((n_tasks, STREAM), dtype=np.int64)
    state = rs.randint(VOCAB, size=n_tasks)
    rows = np.arange(n_tasks)
    for t in range(STREAM):
        tokens[:, t] = state
        u = rs.random(n_tasks)[:, None]
        state = (P[rows, state].cumsum(-1) > u).argmax(-1)
    return jnp.asarray(tokens, dtype=jnp.int32)


def rnn_init(key: jax.Array):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(embed=0.3 * jax.random.normal(k1, (VOCAB, HIDDEN)),
                wh=0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN),
                out=0.1 * jax.random.normal(k3, (HIDDEN, VOCAB)))


def rnn_apply(w, h, token):
    h = jnp.tanh(w['embed'][token] + w['wh'] @ h)
    return h, h @ w['out']


def forecast_sequence(theta, tokens: jax.Array):
    """One task: scan the fast-weight cell down the token sequence."""
    def cell(carry, token):
        # the carry TRIPLE: fast weights, hidden state, and the previous
        # token, which is the next_step register spelled by hand
        w, h, prev = carry

        def selfsup(wf):
            h2, logits = rnn_apply(wf, h, prev)
            return -jax.nn.log_softmax(logits)[token], (h2, logits)

        grads, (h2, logits) = jax.grad(selfsup, has_aux=True)(w)
        w2 = jax.tree.map(lambda wi, g, lr: wi - lr * g, w, grads, theta['lr'])
        return (w2, h2, token), logits

    start = (theta['init'], jnp.zeros(HIDDEN), tokens[0])   # primed register
    (_, _, _), logits = jax.lax.scan(cell, start, tokens)
    return logits


def query_xent(theta, tokens: jax.Array) -> jax.Array:
    logits = jax.vmap(lambda s: forecast_sequence(theta, s))(tokens)
    logp = jax.nn.log_softmax(logits)
    picked = jnp.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[:, QUERY0:])


def main() -> None:
    init = rnn_init(jax.random.PRNGKey(0))
    theta = dict(init=init, lr=jax.tree.map(lambda x: jnp.full_like(x, TTT_LR0), init))
    opt = optax.adam(META_LR)

    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    sequence = train.reshape(META_STEPS, TASKS, -1)

    def meta_step(carry, batch):
        theta, opt_state = carry
        loss, grads = jax.value_and_grad(query_xent)(theta, batch)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return (optax.apply_updates(theta, updates), opt_state), loss

    (theta, _), losses = jax.lax.scan(meta_step, (theta, opt.init(theta)), sequence)

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    n = sum(x.size for x in jax.tree.leaves(theta))
    print(f'ttt-rnn by hand: weights={n} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query xent {query_xent(theta, tasks):.2f}')


if __name__ == '__main__':
    main()
