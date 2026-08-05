"""Population training in one combinator line: train_step(ensemble(...)).

ensemble stacks n independently drawn members under one vmap; train_step
sees one parametric node. The composition trains the whole population in
a single scanned loop — one XLA program, n models — and the population
is ordinary data afterwards: slice out any member and bind it.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import node_def, ensemble, train_step
from nodejax.examples import mse, tile

N_IN, HIDDEN, POP = 4, 16, 8


def mlp_def():
    def param(rng):
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (N_IN, HIDDEN)),
                      w2=0.5 * jax.random.normal(rng.next(), (HIDDEN, 1)))

    def apply(param, input):
        return jnp.tanh(input @ param.w1) @ param.w2

    return node_def(apply, param=param, name='mlp')


def test_population_trains_in_one_line():
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    population = ensemble(mlp_def(), n=POP)
    trainer = train_step(population, mse, optax.adam(1e-2))

    # one key draws POP independent models; one scan trains them all
    state = trainer.init(model=population.parameterize(rng=jax.random.PRNGKey(1)).param)
    steps = 500
    state, losses = trainer.scan(state, Struct(input=tile(X, steps), target=tile(y, steps)))
    assert losses[-1] < 0.05                       # the population fits

    # the population is data: per-member fit, best member sliced out and
    # bound as an ordinary single model
    outs = population.apply(state.model, X)        # (POP, 64, 1)
    per_member = jnp.mean((outs - y) ** 2, axis=(1, 2))
    assert per_member.shape == (POP,)
    assert jnp.unique(per_member).size == POP      # independent draws, distinct fits

    best = jax.tree.map(lambda w: w[jnp.argmin(per_member)], state.model)
    champion = mlp_def().bind(best)
    assert jnp.allclose(mse(champion.apply(X), y), jnp.min(per_member), atol=1e-6)
