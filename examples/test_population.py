"""Population training via `train_step(ensemble(...))`.

`ensemble` maps `n` independently parameterized copies of a `Node` under a
single `jax.vmap`, exposing one combined parameter tree with leading axis `n`.
Wrapping that population in `train_step` and `trained` optimizes all `n`
members in a single scanned XLA loop, after which indexing any row along axis 0
of `done.param` extracts an individual model's parameters for standalone
binding.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, trained, ensemble, train_step, nn, tile
from examples.util import mse
N_IN, HIDDEN, POP = 4, 16, 8


def MLP() -> Node:
    """Builds a two-layer `tanh` regression network with hidden width `HIDDEN`."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(1)


def test_population_trains_in_one_line() -> None:
    """Verifies parallel training of an `ensemble` population and standalone binding of the lowest-loss member."""
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    # A single root key initializes `POP` independent models, and `trained` optimizes them jointly.
    population = ensemble(MLP().with_input(X), n=POP).parameterize(
        rng=jax.random.PRNGKey(1)).initialize()
    trainer = train_step(population, mse, optax.adam(1e-2))
    steps = 3
    done, aux = trained(trainer).apply(input=tile(X, steps),
                                       target=tile(y, steps))
    assert aux.loss.shape == (steps,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, done.param, population.param))

    # Indexing the mapped parameter axis extracts the best member's weights to bind onto `MLP()`.
    outs = done(X)
    per_member = jnp.mean((outs - y) ** 2, axis=(1, 2))
    assert per_member.shape == (POP,)
    assert jnp.unique(per_member).size == POP

    best = jax.tree.map(lambda w: w[jnp.argmin(per_member)], done.param)
    champion = MLP().bind(best)
    assert jnp.allclose(mse(champion.apply(X), y), jnp.min(per_member), atol=1e-6)
