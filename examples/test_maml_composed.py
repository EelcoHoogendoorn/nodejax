"""Model-agnostic meta-learning (MAML) assembled directly from two `train_step` Nodes.

`Adapted` wraps a support-set `trained(trainer.pnode)` run together with a single
query evaluation, exposing the inner model's initial parameters (and any optimizer
parameters) as the wrapper's own `param` tree:

    inner   train_step(model, loss, sgd)         one inner adaptation step
    adapt   Adapted(inner)                       support scan from `param`, then query call
    outer   train_step(batch(adapt), loss, adam) one meta-optimization step

Because the inner rollout differentiates with respect to the initialization stored
in `param`, wrapping `batch(adapt)` in a second `train_step` differentiates through
inner adaptation without dedicated meta-learning primitives. `test_finetune.py`
verifies the same analytic problem through the stock `finetune` transform.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    node, batch, train_step, trained, Wrapper, PNode, PSNode,
)
from nodejax.struct import Struct
from nodejax.control import Gain
from nodejax import tile


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean squared error on query predictions."""
    return jnp.mean((pred - target) ** 2)


@node
def Adapted(trainer: PSNode) -> PNode:
    """Adapts `trainer` across a support episode from its parameter initialization and evaluates on `query`.

    Wrapping `trained(trainer.pnode)` keeps `trainer`'s parameter tree exposed on the
    returned Node while reinitializing inner optimizer state at the start of every
    episode, allowing an enclosing `train_step` to differentiate through support
    adaptation into the shared initialization.
    """
    if not trainer.state_bound:
        raise TypeError(f'Adapted expects a state-bound trainer; got {trainer!r}')

    # Passing `trainer.pnode` to `trained` reinitializes optimizer state from
    # `param` at the start of each support scan and returns the adapted model.
    run = trained(trainer.pnode)
    model = trainer.members.objective.members.model

    def answer(self, support, query, rng):
        done = self.run(input=support.input, target=support.target)
        _, output = done(query, rng=rng)
        return output

    return Wrapper(run=run)(
        answer, name='adapted',
        rng_from=model.contract.apply_takes_rng)


def test_maml_is_two_train_steps() -> None:
    """Verifies that nested `train_step` composition converges to the analytic MAML optimum.

    For two tasks `a` in `{2, 4}` with three `sgd(0.1)` support steps, the
    post-adaptation meta-loss `0.8^6 * ((theta - 3)^2 + 1)` attains its minimum
    `0.2621` at `theta = 3`.
    """
    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    adapt = Adapted(train_step(model, mse, optax.sgd(0.1)))
    trainer = train_step(batch(adapt).initialize(), mse, optax.adam(0.1))

    a = jnp.array([2.0, 4.0])
    k = 3
    tasks = Struct(
        support=Struct(input=jnp.ones((2, k)), target=jnp.tile(a[:, None], (1, k))),
        query=jnp.ones(2))

    steps = 400
    final, aux = trained(trainer).apply(
        support=tile(tasks.support, steps),
        query=tile(tasks.query, steps),
        target=tile(a, steps),
    )

    # Meta-initialization converges to the analytic task mean.
    meta_init = final.param.objective.model.scale
    assert jnp.allclose(meta_init, 3.0, atol=0.05), meta_init
    assert jnp.allclose(aux.loss[-1], 0.2621, atol=0.01), aux.loss[-1]


def test_meta_sgd_is_the_same_composition_with_a_learned_rate() -> None:
    """Verifies that replacing `optax.sgd` with `learned_sgd` meta-learns inner step sizes without altering `Adapted`."""
    from nodejax.transforms.learning import learned_sgd

    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    adapt = Adapted(train_step(model, mse, learned_sgd(0.1)))
    trainer = train_step(batch(adapt).initialize(), mse, optax.adam(0.1))

    a = jnp.array([2.0, 4.0])
    tasks = Struct(support=Struct(input=jnp.ones((2, 1)), target=a[:, None]),
                   query=jnp.ones(2))

    steps = 300
    final, aux = trained(trainer).apply(
        support=tile(tasks.support, steps),
        query=tile(tasks.query, steps),
        target=tile(a, steps),
    )

    # Outer optimization updates `learned_sgd`'s step-size parameter away from `0.1`.
    rate = final.param.opt.model.scale
    assert not jnp.allclose(rate, 0.1), rate
    assert aux.loss[-1] < aux.loss[0]
