"""Reusable test-time training wrapper in Haiku across stateless and recurrent modules.

Haiku collects parameters ambiently through `hk.transform`, whereas online fast weights have to step as explicit JAX values inside `lax.scan`. Writing a reusable `TTT` `hk.Module` over an arbitrary Haiku callable therefore requires running an inner `hk.transform_with_state` inside the outer `hk.transform`, lifting the inner parameter tree into outer `hk.lift` storage alongside a parallel `hk.get_parameter` rate tree, separating inner `params` from inner recurrent `state`, and threading `(fast_params, fast_state, prev_token)` through `lax.scan` by hand.

Run directly with `python -m examples.comparisons.ttt.ttt_haiku`.
"""

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jax.nn.log_softmax(logits)[target]


def query_xent(preds: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(preds)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[..., QUERY0:])


class MLP(hk.Module):
    """Stateless Haiku token predictor (`token -> logits`)."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__(name='mlp')
        self.vocab = vocab
        self.hidden = hidden

    def __call__(self, token: jax.Array) -> jax.Array:
        embed = hk.get_parameter(
            'embed', (self.vocab, self.hidden), init=hk.initializers.RandomNormal(0.3)
        )
        w = hk.get_parameter(
            'w', (self.hidden, self.hidden),
            init=hk.initializers.RandomNormal(0.3 / float(np.sqrt(self.hidden))),
        )
        out = hk.get_parameter(
            'out', (self.hidden, self.vocab), init=hk.initializers.RandomNormal(0.1)
        )
        return jnp.tanh(embed[token] @ w) @ out


class RNN(hk.Module):
    """Stateful Haiku token predictor storing hidden carry via `hk.get_state`."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__(name='rnn')
        self.vocab = vocab
        self.hidden = hidden

    def __call__(self, token: jax.Array) -> jax.Array:
        embed = hk.get_parameter(
            'embed', (self.vocab, self.hidden), init=hk.initializers.RandomNormal(0.3)
        )
        wh = hk.get_parameter(
            'wh', (self.hidden, self.hidden),
            init=hk.initializers.RandomNormal(0.3 / float(np.sqrt(self.hidden))),
        )
        out = hk.get_parameter(
            'out', (self.hidden, self.vocab), init=hk.initializers.RandomNormal(0.1)
        )
        h_prev = hk.get_state('h', (self.hidden,), init=jnp.zeros)
        h = jnp.tanh(embed[token] + wh @ h_prev)
        hk.set_state('h', h)
        return h @ out


class TTT(hk.Module):
    """Reusable online gradient-adaptation wrapper over any single-token Haiku callable."""

    def __init__(self, step_fn: Callable[[jax.Array], jax.Array],
                 loss_fn: Callable, lr0: float):
        super().__init__(name='ttt')
        self.inner = hk.without_apply_rng(hk.transform_with_state(step_fn))
        self.loss_fn = loss_fn
        self.lr0 = lr0

    def __call__(self, tokens: jax.Array) -> jax.Array:
        lift_params = hk.lift(
            lambda k, x: self.inner.init(k, x)[0], name='inner'
        )
        key = hk.next_rng_key() if hk.running_init() else jax.random.PRNGKey(0)
        params0 = lift_params(key, tokens[0])
        _, state0 = self.inner.init(jax.random.PRNGKey(0), tokens[0])
        rates = jax.tree.map_with_path(
            lambda path, leaf: hk.get_parameter(
                'lr_' + '_'.join(str(getattr(p, 'key', p)) for p in path),
                leaf.shape,
                init=hk.initializers.Constant(self.lr0),
            ),
            params0,
        )

        def step(carry, token: jax.Array):
            fast_p, fast_s, prev = carry

            def selfsup(p):
                out, next_s = self.inner.apply(p, fast_s, prev)
                return self.loss_fn(out, token), (next_s, out)

            grads, (next_s, out) = jax.grad(selfsup, has_aux=True)(fast_p)
            next_p = jax.tree.map(lambda w, g, r: w - r * g, fast_p, grads, rates)
            return (next_p, next_s, token), out

        _, preds = jax.lax.scan(step, (params0, state0, tokens[0]), tokens)
        return preds


def run_variant(variant: str, make_inner: Callable[[], Callable[[jax.Array], jax.Array]],
                tokens: jax.Array, held_out: jax.Array) -> None:
    def batched_fn(batch: jax.Array) -> jax.Array:
        ttt = TTT(make_inner(), xent, TTT_LR0)
        return hk.vmap(ttt, split_rng=False)(batch)

    transformed = hk.without_apply_rng(hk.transform(batched_fn))
    params0 = transformed.init(jax.random.PRNGKey(0), tokens[0])
    preds0 = transformed.apply(params0, held_out)

    opt = optax.adam(META_LR)

    @jax.jit
    def fit(p0, seq):
        def meta_step(carry, batch):
            p, opt_s = carry
            loss, grads = jax.value_and_grad(
                lambda cur_p: query_xent(transformed.apply(cur_p, batch), batch)
            )(p)
            updates, next_opt = opt.update(grads, opt_s, p)
            return (optax.apply_updates(p, updates), next_opt), loss

        (done_p, _), losses = jax.lax.scan(meta_step, (p0, opt.init(p0)), seq)
        return done_p, losses

    params1, losses = fit(params0, tokens)
    preds1 = transformed.apply(params1, held_out)
    n_weights = sum(x.size for x in jax.tree.leaves(params1))
    report(
        'haiku', variant, n_weights, bool(jnp.all(jnp.isfinite(losses))),
        query_nll_numpy(preds0, held_out), query_nll_numpy(preds1, held_out),
    )


def main() -> None:
    tokens = jnp.asarray(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1)
    )
    held_out = jnp.asarray(make_tasks(np.random.RandomState(99), EVAL_TASKS))
    run_variant('mlp', lambda: (lambda tok: MLP(VOCAB, HIDDEN)(tok)), tokens, held_out)
    run_variant('rnn', lambda: (lambda tok: RNN(VOCAB, HIDDEN)(tok)), tokens, held_out)


if __name__ == '__main__':
    main()
