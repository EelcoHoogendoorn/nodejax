"""Population training in one combinator line: train_step(ensemble(...)).

ensemble stacks n independently drawn members under one vmap; train_step
sees one parametric node. The composition trains the whole population in
a single scanned loop — one XLA program, n models — and the population
is ordinary data afterwards: index any member's row out of the param
tree and bind it.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import Node, trained, ensemble, train_step, nn
from nodejax import tile
from nodejax.examples.util import mse
N_IN, HIDDEN, POP = 4, 16, 8


def MLP() -> Node:
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(1)   # sizes flow from the data


def test_population_trains_in_one_line():
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    # one key draws POP independent models; one finalized run trains them
    # all, and what comes back IS the trained population, callable
    population = ensemble(MLP().with_input(X), n=POP).parameterize(
        rng=jax.random.PRNGKey(1)).initialize()
    trainer = train_step(population, mse, optax.adam(1e-2))
    steps = 500
    done, aux = trained(trainer).apply(input=tile(X, steps),
                                       target=tile(y, steps))
    assert aux.loss[-1] < 0.05                       # the population fits

    # the population is data: per-member fit, the best member's row
    # INDEXED out of the mapped param axis (ensemble refuses the member
    # slice door for exactly this reason: a row is an indexing, not a
    # slot) and bound as an ordinary single model
    _, outs = done(X)                                # (POP, 64, 1)
    per_member = jnp.mean((outs - y) ** 2, axis=(1, 2))
    assert per_member.shape == (POP,)
    assert jnp.unique(per_member).size == POP      # independent draws, distinct fits

    best = jax.tree.map(lambda w: w[jnp.argmin(per_member)], done.param)
    champion = MLP().bind(best)
    assert jnp.allclose(mse(champion.apply(X), y), jnp.min(per_member), atol=1e-6)
