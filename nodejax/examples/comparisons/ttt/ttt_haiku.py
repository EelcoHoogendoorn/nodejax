"""ttt_nodejax's ttt-rnn row, the haiku side of the framework comparison.

Self-contained: no nodejax import, the same model, task family, budget and
scoring as ttt_rnn_by_hand.

What the column prices. Haiku's meta-params are AMBIENT, reached by
hk.get_parameter under a path the call stack assigns, so the initialization
and the per-leaf rates are declared where they are used and collected by the
transform. The fast weights are the interesting boundary: they start FROM
the ambient parameters but cannot remain ambient, because a value that
changes per step and takes gradients has to be a value, so the first line of
the forecast copies the parameters out into an ordinary dict and everything
after is the by-hand file: plain jax.grad over a pure closure, a lax.scan
carrying (weights, hidden, previous token), hk.vmap over tasks. The pairing
rides the carry by hand, as in every rival.

The transform boundary is where the two-level structure shows. Inside, the
inner adaptation differentiates a closure over VALUES; outside, the meta
loop differentiates f.apply over the collected parameters. Neither level
knows the other exists, which is the same statement nodejax makes with two
train_steps, made here by one function crossing hk.transform.

Run directly:  python -m nodejax.examples.comparisons.ttt.ttt_haiku
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
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


def rnn_apply(w, h, token):
    h = jnp.tanh(w['embed'][token] + w['wh'] @ h)
    return h, h @ w['out']


def batched_forecast(tokens: jax.Array):
    """The whole batched rollout, one haiku function. The meta-params are
    ambient; the fast weights copy out of them and become the scan carry."""
    init = dict(
        embed=hk.get_parameter('embed', (VOCAB, HIDDEN),
                               init=hk.initializers.RandomNormal(0.3)),
        wh=hk.get_parameter('wh', (HIDDEN, HIDDEN),
                            init=hk.initializers.RandomNormal(0.3 / np.sqrt(HIDDEN))),
        out=hk.get_parameter('out', (HIDDEN, VOCAB),
                             init=hk.initializers.RandomNormal(0.1)))
    rates = {name: hk.get_parameter(f'lr_{name}', leaf.shape,
                                    init=hk.initializers.Constant(TTT_LR0))
             for name, leaf in init.items()}

    def forecast_one(sequence):
        def cell(carry, token):
            # the previous token is the pairing, threaded by hand where
            # the nodejax row spells it as a next_step register
            w, h, prev = carry

            def selfsup(wf):
                h2, logits = rnn_apply(wf, h, prev)
                return -jax.nn.log_softmax(logits)[token], (h2, logits)

            grads, (h2, logits) = jax.grad(selfsup, has_aux=True)(w)
            # the inner rule is WELDED into the loop body, as in every rival:
            # swapping it means changing the carry and the parameter set by
            # hand, where the nodejax row swaps an argument
            w2 = jax.tree.map(lambda wi, g, lr: wi - lr * g, w, grads, rates)
            return (w2, h2, token), logits

        start = (init, jnp.zeros(HIDDEN), sequence[0])        # primed register
        (_, _, _), logits = jax.lax.scan(cell, start, sequence)
        return logits

    return hk.vmap(forecast_one, split_rng=False)(tokens)


def main() -> None:
    f = hk.without_apply_rng(hk.transform(batched_forecast))

    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    sequence = train.reshape(META_STEPS, TASKS, -1)
    params = f.init(jax.random.PRNGKey(0), sequence[0])
    opt = optax.adam(META_LR)

    def query_xent(params, batch):
        logits = f.apply(params, batch)
        logp = jax.nn.log_softmax(logits)
        picked = jnp.take_along_axis(logp, batch[..., None], axis=-1)[..., 0]
        return -jnp.mean(picked[:, QUERY0:])

    def meta_step(carry, batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(query_xent)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss

    (params, _), losses = jax.lax.scan(meta_step, (params, opt.init(params)), sequence)

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    n = sum(x.size for x in jax.tree.leaves(params))
    print(f'ttt-rnn haiku: weights={n} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query xent {query_xent(params, tasks):.2f}')


if __name__ == '__main__':
    main()
