"""Test-time training, the nodejax side of the framework comparison.

The row: a tanh rnn whose weights adapt EVERY STEP by one gradient
step on the self-supervised next-step loss, while its hidden state
carries across steps — two memories at two speeds in one cell. Here
that is one composition line, scan(ttt(rnn)), because ttt is a generic
wrapper: it demotes ANY node's params to per-step state and makes the
state transition one gradient step. The rival files implement the SAME
row against the same task family and budget: ttt_rnn_by_hand (raw jax,
the ground truth), ttt_rnn_flax (nnx) and ttt_rnn_torch — the
comparison is what each framework charges for the generic wrapper.
ttt_variants.py runs the wider study (fixed baselines and other inner
models) on the same harness.

Every row consumes the train_step-style stream
Struct(input=<L lags>, target=<the next value>). For the ttt rows this
IS the self-supervision — the target column is derived from the
stream itself, and ttt's predict-then-update order keeps scoring
prequential (every emitted prediction comes from weights that have not
trained on its target). Supervised rows read the same columns as
labeled data.

Writes plots/meta_<name>.png per row and a summary line to stdout.

Run directly:  python -m nodejax.examples.comparisons.ttt_nodejax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import NodeDef, node_def, scan, batch, ttt, train_step, KeyStream
from nodejax.struct import Struct

from nodejax.examples.comparisons.ttt_common import (
    LAGS, HIDDEN, TASKS, META_STEPS, TTT_LR0, META_LR,
    make_tasks, mse, query_mse, report)


def RNN(width: int, hidden: int) -> NodeDef:
    """Recurrent predictor over width-vector inputs: the hidden state
    advances on the lags — each stream value enters the recurrence as
    it ages into one. The input width is the caller's to state: it is
    a property of the data, not of the cell. It is an argument rather
    than an ndef.input read because an outer with_input does not yet
    propagate through scan/ttt to a shape-reading constructor (the
    spec-propagation gap noted in nodejax/transforms/common.py)."""
    def param(rng: KeyStream) -> Struct:
        return Struct(win=0.5 * jax.random.normal(rng.next(), (width, hidden)),
                      wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
                      b=jnp.zeros(hidden),
                      wout=0.3 * jax.random.normal(rng.next(), (hidden,)))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.b.shape)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input @ param.win + param.wh @ state + param.b)
        return h, param.wout @ h

    return node_def(apply, init=init, param=param, name='rnn')


def model_ttt(predictor: NodeDef, lr0: float) -> NodeDef:
    return scan(ttt(predictor, mse, lr0))                # weights adapt down the stream


def feed_samples(tasks):
    """The ttt rows' model input: the task struct as-is — make_tasks
    already emits train_step's sample stream, targets included,
    because the ttt cell trains as it goes."""
    return tasks


def run_stream(name: str, model: NodeDef, feed) -> None:
    """Meta-train a row on the shared budget, evaluate on held-out
    tasks, report. feed picks the model's input off the task struct."""
    trainer = train_step(batch(model), query_mse, optax.adam(META_LR))
    m = batch(model).parameterize(rng=jax.random.PRNGKey(0))
    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    final, losses = trainer.scan(trainer.init(model=m.param), Struct(input=jax.tree.map(fold, feed(train)), target=fold(train.target)))

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    out = batch(model).apply(final.model, feed(tasks))
    n = sum(x.size for x in jax.tree.leaves(final.model))
    report(name, n, bool(jnp.all(jnp.isfinite(losses))), out, tasks)


def main() -> None:
    run_stream('ttt-rnn', model_ttt(RNN(LAGS, HIDDEN), TTT_LR0), feed_samples)


if __name__ == '__main__':
    main()
