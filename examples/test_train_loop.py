"""Host-side training control around a scanned `train_step` Node.

`train_step` defines one pure, resumable optimizer update, and `scan(train_step(...))`
fuses a chunk of updates into one compiled state-passing call. Metric aggregation,
logging, and early stopping execute in ordinary Python between chunks:

    for chunk in chunks:
        chunk_trainer, (_, aux) = chunk_trainer(bundle=chunk)
        log.append(stats(chunk_trainer.params(), aux))
        if plateaued(log):
            break

Each call returns a successor `PSNode` without mutating the prior value, allowing
the caller to inspect parameters and `Aux` records, log metrics, or replay a
chunk deterministically from any saved trainer state.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Aux, Node, Struct, Wrapper, node, scan, tile, train_step
from nodejax import nn
from examples.util import mse
CHUNK = 2
MAX_CHUNKS = 3


@node
def PredictionStats(model: Node) -> Node:
    """Wraps `model` to emit root-mean-square prediction magnitude in `Aux` without changing primary output."""
    def apply(self, input):
        prediction = self.model(input)
        prediction_rms = jnp.sqrt(jnp.mean(prediction**2))
        return prediction, Aux(prediction_rms=prediction_rms)

    return Wrapper(model=model)(apply)


def test_chunked_loop_with_host_side_stats() -> None:
    """Verifies host-side `Aux` logging, early stopping, and deterministic replay across scanned `train_step` chunks."""
    w_true = jnp.array([[1.5], [-2.0], [0.5], [3.0]])
    X = jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    y = X @ w_true

    initial = scan(train_step(
        PredictionStats(nn.Linear(1)).with_input(X),
        mse,
        optax.sgd(0.1),
    )).parameterize(rng=jax.random.PRNGKey(1)).initialize()
    chunk_trainer = initial
    chunk_x, chunk_y = tile(X, CHUNK), tile(y, CHUNK)

    log = []
    for i in range(MAX_CHUNKS):
        # Each call executes `CHUNK` compiled updates and returns the successor trainer `PSNode`.
        chunk_trainer, (_, aux) = chunk_trainer(input=chunk_x, target=chunk_y)

        # Nested `Aux` fields surface alongside `aux.loss` as ordinary `Struct` attributes.
        log.append(Struct(
            chunk=i,
            loss=float(jnp.mean(aux.loss)),
            prediction_rms=float(jnp.mean(aux.objective.model.prediction_rms)),
        ))

        if len(log) >= 2 and log[-1].prediction_rms > 0.0:
            break

    assert len(log) < MAX_CHUNKS
    assert all(jnp.isfinite(entry.loss) for entry in log)
    assert log[-1].prediction_rms > 0.0
    assert not jax.tree.all(jax.tree.map(
        jnp.array_equal, chunk_trainer.params(), initial.params()))
    # Re-evaluating the same `PSNode` value reproduces identical chunk losses.
    _, (_, again) = chunk_trainer(input=chunk_x, target=chunk_y)
    _, (_, once_more) = chunk_trainer(input=chunk_x, target=chunk_y)
    assert jnp.allclose(again.loss, once_more.loss)
