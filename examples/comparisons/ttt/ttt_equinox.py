"""ttt_nodejax's ttt-rnn row, the equinox side of the framework comparison.

Self-contained: no nodejax import, the same model, task family, budget and
scoring as ttt_rnn_by_hand.

What the column prices. Equinox's one idea, a module IS a pytree, is exactly
the ttt trick's shape: weights carried as a value. So the fast learner is
literally the module, threaded through lax.scan and stepped by gradient, and
the meta-params are an ordinary (model, rates) pair of pytrees, the rates a
module-shaped tree of step sizes. eqx.filter_grad differentiates with
respect to a module, eqx.apply_updates folds updates into one: the framework
contributes the pytree discipline and those two calls.

What it does not contribute is any structure for the loops: the token scan,
the task vmap and the meta scan are raw jax around the module, the same
skeleton as the by-hand file, and the previous token rides the scan carry by
hand where the nodejax row spells it as a next_step register. The module
system never sees the composition, only the leaves.

Run directly:  python -m examples.comparisons.ttt.ttt_equinox
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
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


class RNN(eqx.Module):
    """The inner learner. A module is a pytree of its arrays, so 'weights as
    fast state' needs no adapter: the instance itself is the scan carry."""

    embed: jax.Array
    wh: jax.Array
    out: jax.Array

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embed = 0.3 * jax.random.normal(k1, (VOCAB, HIDDEN))
        self.wh = 0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN)
        self.out = 0.1 * jax.random.normal(k3, (HIDDEN, VOCAB))

    def __call__(self, h, token):
        h = jnp.tanh(self.embed[token] + self.wh @ h)
        return h, h @ self.out


def forecast_sequence(model, rates, tokens: jax.Array):
    """One task: the fast module, its hidden and the previous token as the
    scan carry. Predict-then-update, one gradient step per token at the
    meta-learned per-leaf rates."""
    def cell(carry, token):
        # the previous token is the pairing, threaded by hand where the
        # nodejax row spells it as a next_step register
        fast, h, prev = carry

        def selfsup(m):
            h2, logits = m(h, prev)
            return -jax.nn.log_softmax(logits)[token], (h2, logits)

        grads, (h2, logits) = eqx.filter_grad(selfsup, has_aux=True)(fast)
        # the inner rule is WELDED into the loop body: per-leaf learned-rate
        # sgd exists in no stock optax, so it is arithmetic here, and
        # swapping it (momentum, adam) means changing the carry to thread
        # the rule's state and the meta tree to hold its params, by hand.
        # The nodejax row makes the same choice an argument
        fast = eqx.apply_updates(
            fast, jax.tree.map(lambda g, lr: -lr * g, grads, rates))
        return (fast, h2, token), logits

    start = (model, jnp.zeros(HIDDEN), tokens[0])           # primed register
    (_, _, _), logits = jax.lax.scan(cell, start, tokens)
    return logits


def query_xent(theta, tokens: jax.Array) -> jax.Array:
    model, rates = theta
    logits = jax.vmap(lambda s: forecast_sequence(model, rates, s))(tokens)
    logp = jax.nn.log_softmax(logits)
    picked = jnp.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[:, QUERY0:])


def main() -> None:
    model = RNN(jax.random.PRNGKey(0))
    theta = (model, jax.tree.map(lambda x: jnp.full_like(x, TTT_LR0), model))
    opt = optax.adam(META_LR)

    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    sequence = train.reshape(META_STEPS, TASKS, -1)

    def meta_step(carry, batch):
        theta, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(query_xent)(theta, batch)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return (eqx.apply_updates(theta, updates), opt_state), loss

    (theta, _), losses = jax.lax.scan(meta_step, (theta, opt.init(theta)), sequence)

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    n = sum(x.size for x in jax.tree.leaves(theta))
    print(f'ttt-rnn equinox: weights={n} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query xent {query_xent(theta, tasks):.2f}')


if __name__ == '__main__':
    main()
