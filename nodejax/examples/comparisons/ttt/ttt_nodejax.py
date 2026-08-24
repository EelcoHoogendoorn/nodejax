"""Test-time training on next-token prediction, the nodejax side of the
framework comparison.

The row: a recurrent next-token model whose weights adapt EVERY STEP by
one gradient step on the token it just failed to predict, while its
hidden state carries beneath them: two memories at two speeds in one
cell. Here that is one composition line,
scanned(next_step_ttt(train_step(model, xent, learned_sgd(lr0)))),
because the gradient cell is generic over its model and the next_step
register assembles the (previous token, this token) pair in-graph. The
rival files implement the SAME row against the same task family and
budget: ttt_rnn_by_hand (raw jax, the ground truth), ttt_rnn_flax
(nnx), ttt_rnn_torch, ttt_equinox and ttt_haiku. The comparison is what
each framework charges for the generic wrapper, and one structural row
is the pairing itself: every rival threads the previous token through
its scan by hand.

HOW THE STREAM IS FED. The outer trainer imposes its element contract
Struct(input=..., target=...), and both fields are the SAME raw token
sequence: the register derives each step's pair from `input`, and
query_xent scores the emitted logits against `target` over the query
region. No column of derived data exists anywhere; the loader hands
over tokens and nothing else. Levels above peel leading axes and impose
nothing new:

    outer scan   (META_STEPS, ...)      one meta step per element
    batch        (TASKS, ...)           one task per batch element
    scanned      (STREAM, ...)          one token per inner step
    ttt cell     one (previous, current) pair from the register

Writes plots/meta_<name>.png per row and a summary line to stdout.

Run directly:  python -m nodejax.examples.comparisons.ttt.ttt_nodejax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax.transforms.train_step import learned_sgd
from nodejax import (node, trained, Node, Leaf, scan, scanned, batch,
                     train_step, next_step_ttt, KeyStream)
from nodejax.struct import Struct

from nodejax.examples.comparisons.ttt.ttt_common import (
    VOCAB, HIDDEN, TASKS, META_STEPS, TTT_LR0, META_LR,
    make_tasks, xent, query_xent, report)


@node
def RNN(vocab: int, hidden: int) -> Node:
    """Recurrent next-token predictor: embed the previous token, advance
    the hidden state, emit logits for the current one."""
    def param(rng: KeyStream) -> Struct:
        return Struct(embed=0.3 * jax.random.normal(rng.next(), (vocab, hidden)),
                      wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
                      out=0.1 * jax.random.normal(rng.next(), (hidden, vocab)))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.wh.shape[0])

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(param.embed[input] + param.wh @ state)
        return h, h @ param.out

    return Leaf(apply, init=init, param=param)


def ttt_next_token() -> Node:
    """The whole row, three lines. The INNER train_step demotes the
    model's weights to per-step state, adapting down the sequence one
    gradient step per token at meta-learned per-leaf rates (Meta-SGD's
    device, applied at test time); the next_step register makes each
    token the target of the one before it, so the raw sequence is the
    only data anywhere. The OUTER train_step, batched over tasks,
    learns the initialization and rates the inner adapts from, its
    gradient travelling through the inner run."""
    cell = next_step_ttt(train_step(RNN(VOCAB, HIDDEN), xent, learned_sgd(TTT_LR0)))
    model = scanned(cell)
    return train_step(batch(model), query_xent, optax.adam(META_LR))


def run_sequence(name: str, trainer) -> None:
    """Meta-train on the shared budget, evaluate on held-out tasks,
    report. The sequence is the raw token matrix twice: `input` feeds the
    register, `target` feeds the meta-objective, and train_step's
    element contract is why the same array appears under two names."""
    tokens = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    folded = tokens.reshape(META_STEPS, TASKS, -1)
    final, aux = trained(trainer).apply(input=folded, target=folded)

    held_out = make_tasks(np.random.RandomState(99), TASKS)
    # trained's product IS the model, state-bound: call it as it stands
    _, predictions = final(held_out)
    weight_count = sum(leaf.size for leaf in jax.tree.leaves(final.param))
    report(name, weight_count, bool(jnp.all(jnp.isfinite(aux.loss))),
           predictions, held_out)


def main() -> None:
    trainer = ttt_next_token().parameterize(rng=jax.random.PRNGKey(0))
    run_sequence('ttt-rnn', trainer)


if __name__ == '__main__':
    main()
