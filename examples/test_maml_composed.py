"""MAML with both training loops in plain sight.

`finetune` and `metasgd` are names for a composition, and this is the
composition. Nothing here is a transform of its own: the inner loop is a
train_step run over a support set, the outer loop is another train_step, and
the only hand-written line is the one that answers the query.

    inner   train_step(model, loss, sgd)        one adaptation step
    adapt   the inner trainer scanned over the support set, restarted
            from its param each episode
    ANSWER  apply the tuned weights to the query
    outer   train_step(batch(adapt), loss, adam)   one meta step

The two train_steps are the two loops of the algorithm, and they are the same
node used twice. What makes the outer one meta is that its gradient travels
through the inner one, which needs nothing said: the inner loop is an
ordinary differentiable function of the initialization it started from, and
that initialization is the outer loop's param.

Read beside test_finetune.py, which asserts the same numbers through
`finetune`. The question this file exists to answer is what that transform is
worth: everything below except `answer` is boilerplate the transform would
have written, and `answer` is two lines.
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
    """Score the clean query output, excluding inner training telemetry."""
    return jnp.mean((pred - target) ** 2)


@node
def Adapted(trainer: PSNode) -> PNode:
    """Run an inner train-step node over support, then answer the query with
    its tuned model.

    The wrapper keeps the train-step definition as its member and promotes
    its param, which means the initialization this
    adapts FROM is an ordinary param of an ordinary node, and that is the
    whole reason the outer loop can learn it: nothing here is meta, and
    the outer train_step is the same node as the inner one. Each episode
    restarts from the train step's own reset, rebuilt from param, so the
    meta gradient reaches the start of every inner run."""
    if not trainer.state_bound:
        raise TypeError(f'Adapted expects a state-bound trainer; got {trainer!r}')

    # Drop the trainer's arrived state: every episode must restart from the
    # initialization held in its params. ``trained`` on the PNode rung already
    # means exactly “reset, scan the support set, return the finished model”.
    # The wrapper contributes only the query that model answers.
    run = trained(trainer.pnode)
    model = trainer.members.model

    def answer(self, support, query, rng):
        done = self.run(input=support.input, target=support.target)
        _, output = done(query, rng=rng)
        return output

    return Wrapper(run=run)(
        answer, name='adapted',
        rng_from=model.contract.apply_takes_rng)


def test_maml_is_two_train_steps():
    """Two tasks, a in {2, 4}. Post-adaptation meta-loss is
    0.8^6 * ((theta-3)^2 + 1), minimized at theta = 3 with value 0.2621.

    The same numbers test_finetune.py asserts through the transform, reached
    here by composition. Converging at all requires the gradient to travel
    THROUGH the inner loop, which the contract gives for nothing."""
    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()  # where it starts
    adapt = Adapted(train_step(model, mse, optax.sgd(0.1)))          # the INNER loop
    trainer = train_step(batch(adapt).initialize(), mse, optax.adam(0.1))  # the OUTER one

    a = jnp.array([2.0, 4.0])
    k = 3
    tasks = Struct(
        support=Struct(input=jnp.ones((2, k)), target=jnp.tile(a[:, None], (1, k))),
        query=jnp.ones(2))

    steps = 400
    final, aux = trained(trainer).apply(input=tile(tasks, steps),
                                  target=tile(a, steps))

    # the meta-init converges to the analytic optimum, the task mean
    meta_init = final.param.model.scale
    assert jnp.allclose(meta_init, 3.0, atol=0.05), meta_init
    assert jnp.allclose(aux.loss[-1], 0.2621, atol=0.01), aux.loss[-1]


def test_meta_sgd_is_the_same_composition_with_a_learned_rate():
    """metasgd is this with the optimizer's step sizes on its param.

        Nothing about the composition changes: one member of the inner train step is
    swapped for an optimizer that carries params, and the outer loop learns them
    because they are params of the node it is training. That is the whole of
    what a separate `metasgd` transform was for."""
    from nodejax.transforms.learning import learned_sgd

    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    adapt = Adapted(train_step(model, mse, learned_sgd(0.1)))
    trainer = train_step(batch(adapt).initialize(), mse, optax.adam(0.1))

    a = jnp.array([2.0, 4.0])
    tasks = Struct(support=Struct(input=jnp.ones((2, 1)), target=a[:, None]),
                   query=jnp.ones(2))

    steps = 300
    final, aux = trained(trainer).apply(input=tile(tasks, steps),
                                  target=tile(a, steps))

    # one step at the learned rate reaches the task from a shared init: the
    # rate moved off where it started, which is the thing being learned
    rate = final.param.opt.scale
    assert not jnp.allclose(rate, 0.1), rate
    assert aux.loss[-1] < aux.loss[0]
