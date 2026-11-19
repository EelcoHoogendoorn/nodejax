"""Reusable test-time training in NodeJAX across stateless and recurrent predictors.

`ttt_meta(model)` wraps any `Node` in three composition steps: `train_step(model, xent, learned_sgd(TTT_LR0))` moves `model`'s parameters into per-token state updated by meta-learned step sizes, `next_step_ttt` pairs each incoming token with the previous token in-graph, and `train_step(batch(scanned(cell)), query_xent, optax.adam(META_LR))` meta-trains across task batches. Passing a stateless `MLP` (`PNode`) or a recurrent `RNN` (`PSNode`) calls the exact same `ttt_meta` function without protocol adapters.

Run directly with `python -m examples.comparisons.ttt.ttt_nodejax`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import (
    KeyStream, Leaf, Node, Struct, batch, next_step_ttt, node,
    scanned, split_aux, train_step, trained,
)
from nodejax.transforms.learning import learned_sgd

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jax.nn.log_softmax(logits)[target]


def query_xent(out: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(out)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[..., QUERY0:])


@node
def MLP(vocab: int, hidden: int) -> Node:
    """Maps each input token to next-token logits without carried state."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            embed=0.3 * jax.random.normal(rng.next(), (vocab, hidden)),
            w=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            out=0.1 * jax.random.normal(rng.next(), (hidden, vocab)),
        )

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return jnp.tanh(param.embed[input] @ param.w) @ param.out

    return Leaf(apply, param=param)


@node
def RNN(vocab: int, hidden: int) -> Node:
    """Carries a recurrent hidden vector alongside fast-weight parameters."""
    def param(rng: KeyStream) -> Struct:
        return Struct(
            embed=0.3 * jax.random.normal(rng.next(), (vocab, hidden)),
            wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            out=0.1 * jax.random.normal(rng.next(), (hidden, vocab)),
        )

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.wh.shape[0])

    def apply(param: Struct, state: jax.Array,
              input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(param.embed[input] + param.wh @ state)
        return h, h @ param.out

    return Leaf(apply, init=init, param=param)


def ttt_meta(model: Node) -> tuple[Node, Node]:
    """Builds both the batched sequence predictor and its outer meta-trainer."""
    cell = next_step_ttt(train_step(model, xent, learned_sgd(TTT_LR0)))
    batched = batch(scanned(cell))
    return batched, train_step(batched, query_xent, optax.adam(META_LR))


def run_variant(variant: str, model: Node, tokens: jax.Array,
                held_out: jax.Array) -> None:
    batched, meta = ttt_meta(model)
    initial = batched.parameterize(rng=jax.random.PRNGKey(0))
    preds0, _ = split_aux(initial.apply(held_out))

    trainer = meta.parameterize(rng=jax.random.PRNGKey(0))
    final, aux = trained(trainer).apply(input=tokens, target=tokens)
    preds1, _ = split_aux(final(held_out))

    n_weights = sum(leaf.size for leaf in jax.tree.leaves(final.param))
    report(
        'nodejax', variant, n_weights, bool(jnp.all(jnp.isfinite(aux.loss))),
        query_nll_numpy(preds0, held_out), query_nll_numpy(preds1, held_out),
    )


def main() -> None:
    tokens = jnp.asarray(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1)
    )
    held_out = jnp.asarray(make_tasks(np.random.RandomState(99), EVAL_TASKS))
    run_variant('mlp', MLP(VOCAB, HIDDEN), tokens, held_out)
    run_variant('rnn', RNN(VOCAB, HIDDEN), tokens, held_out)


if __name__ == '__main__':
    main()
