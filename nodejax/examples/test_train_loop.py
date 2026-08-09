"""The rich training loop is user-land Python: scanned chunks of
train_steps sandwiched between host-side bookkeeping.

train_step gives a pure resumable step; scan compiles a chunk of them
into one XLA call. Everything a 'rich' trainer framework would hook in
— stat collection, logging, early stopping — is a plain Python for
loop BETWEEN chunks, reading the trainer state like any pytree:

    for chunk in chunks:
        state, losses = run_chunk(state, chunk)   # jitted, fast
        log.append(stats(state, losses))          # host, free-form
        if plateaued(log): break                  # host control flow

The trainer state is data, so the host loop can read weights, compute
distances to anything, write to any logger, and stop on any condition;
the node sees none of it. Rich loops are code you write, not framework
features you configure.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import train_step
from nodejax import nn
from nodejax.util import mse, tile
CHUNK = 100          # steps fused into one scan call
MAX_CHUNKS = 10


def test_chunked_loop_with_host_side_stats():
    w_true = jnp.array([[1.5], [-2.0], [0.5], [3.0]])
    X = jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    y = X @ w_true

    lin = nn.Linear(1).with_input(X)
    model = lin.bind(Struct(w=jnp.zeros((4, 1)), b=jnp.zeros(1)))
    trainer = train_step(lin, mse, optax.sgd(0.1))

    state = trainer.init(model=model.param)
    chunk = Struct(input=tile(X, CHUNK), target=tile(y, CHUNK))
    run_chunk = jax.jit(trainer.scan)             # compiled once, reused

    log = []
    for i in range(MAX_CHUNKS):
        state, losses = run_chunk(state, chunk)

        # host side: read the trainer state like any pytree
        log.append(Struct(
            chunk=i,
            loss=float(jnp.mean(losses)),
            w_err=float(jnp.linalg.norm(state.model.w - w_true)),
            w_norm=float(jnp.linalg.norm(state.model.w)),
        ))
        print(f"[loop] chunk {i}: loss {log[-1].loss:.2e} "
              f"|w-w*| {log[-1].w_err:.2e}")

        if log[-1].loss < 1e-8:                   # host control flow
            break

    assert log[-1].loss < 1e-8                    # converged
    assert len(log) < MAX_CHUNKS                  # and stopped early
    assert log[0].loss > 10 * log[-1].loss        # stats saw the descent
    # the loop never broke purity: rerunning the same chunk from the
    # same state reproduces its losses exactly
    state2, again = run_chunk(state, chunk)
    _, once_more = run_chunk(state, chunk)
    assert jnp.allclose(again, once_more)
