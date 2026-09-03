"""The rich training loop is user-land Python around a scanned train step.

``train_step`` defines one pure, resumable update. ``scan(train_step(...))``
turns a chunk of updates into one stateful Node call. Stat collection, logging,
and early stopping remain ordinary host-side Python between those calls:

    for chunk in chunks:
        chunk_trainer, (_, aux) = chunk_trainer(bundle=chunk)
        log.append(stats(chunk_trainer.params(), aux))
        if plateaued(log):
            break

The state-bound chunk trainer is data carried by the host loop. Each call
returns its successor, while the caller reads parameters and Aux, writes to any
logger, and decides when to stop.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Aux, Node, Struct, Wrapper, node, scan, tile, train_step
from nodejax import nn
from examples.util import mse
CHUNK = 100          # steps fused into one scan call
MAX_CHUNKS = 10


@node
def PredictionStats(model: Node) -> Node:
    """Report prediction RMS through Aux without changing model output."""
    def apply(self, input):
        prediction = self.model(input)
        prediction_rms = jnp.sqrt(jnp.mean(prediction**2))
        return prediction, Aux(prediction_rms=prediction_rms)

    return Wrapper(model=model)(apply)


def test_chunked_loop_with_host_side_stats():
    w_true = jnp.array([[1.5], [-2.0], [0.5], [3.0]])
    X = jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    y = X @ w_true

    chunk_trainer = scan(train_step(
        PredictionStats(nn.Linear(1)).with_input(X),
        mse,
        optax.sgd(0.1),
    )).parameterize(rng=jax.random.PRNGKey(1)).initialize()
    chunk_x, chunk_y = tile(X, CHUNK), tile(y, CHUNK)

    log = []
    for i in range(MAX_CHUNKS):
        # One call runs a compiled chunk and returns the trainer state from its end.
        chunk_trainer, (_, aux) = chunk_trainer(input=chunk_x, target=chunk_y)

        # Host side: read the trainer and its nested Aux like ordinary values.
        log.append(Struct(
            chunk=i,
            loss=float(jnp.mean(aux.loss)),
            prediction_rms=float(jnp.mean(aux.objective.model.prediction_rms)),
        ))
        print(f"[loop] chunk {i}: loss {log[-1].loss:.2e} "
              f"| prediction RMS {log[-1].prediction_rms:.2e}")

        if log[-1].loss < 1e-8:                   # host control flow
            break

    assert log[-1].loss < 1e-8                    # converged
    assert log[-1].prediction_rms > 0.0           # internal Aux reached the host log
    assert len(log) < MAX_CHUNKS                  # and stopped early
    assert log[0].loss > 10 * log[-1].loss        # stats saw the descent
    # The trainer is a value, so rerunning from the same value reproduces its losses.
    _, (_, again) = chunk_trainer(input=chunk_x, target=chunk_y)
    _, (_, once_more) = chunk_trainer(input=chunk_x, target=chunk_y)
    assert jnp.allclose(again.loss, once_more.loss)
