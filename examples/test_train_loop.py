"""The rich training loop is user-land Python: scanned chunks of
train_steps sandwiched between host-side bookkeeping.

train_step gives a pure resumable step; scan compiles a chunk of them
into one XLA call. Everything a 'rich' trainer framework would hook in
— stat collection, logging, early stopping — is a plain Python for
loop BETWEEN chunks, reading the trainer state like any pytree:

    for chunk in chunks:
        trainer, (_, aux) = trainer.scan(chunk)   # jitted, fast
        log.append(stats(trainer, aux.loss))      # host, free-form
        if plateaued(log): break                  # host control flow

The training session is one state-bound node, and it is data: the host
loop scans chunks of steps THROUGH it (the session object is the scan
carry), reads its weights like any pytree, writes to any logger, and
stops on any condition; the node sees none of it. Rich loops are code
you write, not framework features you configure.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import train_step
from nodejax import nn
from nodejax import tile
from examples.util import mse
CHUNK = 100          # steps fused into one scan call
MAX_CHUNKS = 10


def test_chunked_loop_with_host_side_stats():
    w_true = jnp.array([[1.5], [-2.0], [0.5], [3.0]])
    X = jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    y = X @ w_true

    # the ladder, end to end: build, parameterize, initialize; the
    # session is one object
    trainer = train_step(
        nn.Linear(1).with_input(X).parameterize(
            rng=jax.random.PRNGKey(1)).initialize(),
        mse, optax.sgd(0.1))
    chunk_x, chunk_y = tile(X, CHUNK), tile(y, CHUNK)

    log = []
    for i in range(MAX_CHUNKS):
        # the loop `trained` does not cover: a chunk of steps scanned
        # THROUGH the session (jitted inside .scan, compiled once), the
        # session crossing calls, the caller owning stats and stopping
        trainer, (_, aux) = trainer.scan(input=chunk_x, target=chunk_y)

        # host side: read the session like any pytree
        log.append(Struct(
            chunk=i,
            loss=float(jnp.mean(aux.loss)),
            w_err=float(jnp.linalg.norm(trainer.state.opt.params.model.w - w_true)),
            w_norm=float(jnp.linalg.norm(trainer.state.opt.params.model.w)),
        ))
        print(f"[loop] chunk {i}: loss {log[-1].loss:.2e} "
              f"|w-w*| {log[-1].w_err:.2e}")

        if log[-1].loss < 1e-8:                   # host control flow
            break

    assert log[-1].loss < 1e-8                    # converged
    assert len(log) < MAX_CHUNKS                  # and stopped early
    assert log[0].loss > 10 * log[-1].loss        # stats saw the descent
    # the loop never broke purity: the trainer is a value, so rerunning
    # the same chunk from the same trainer reproduces its losses exactly
    _, (_, again) = trainer.scan(input=chunk_x, target=chunk_y)
    _, (_, once_more) = trainer.scan(input=chunk_x, target=chunk_y)
    assert jnp.allclose(again.loss, once_more.loss)
