"""Meta-learning as composition.

finetune() is a transform, so MAML is one line:
train_step(batch(finetune(model))) — learning an init that finetunes well.
meta_sgd extends this to optimizer hyperparameters: meta-learning X means
promoting X from a static capture to a component of param.

Task family: y = a * x, model y = scale * x, inner SGD with lr 0.1 on mse.
One inner step contracts the error (scale - a) by (1 - 2*lr), so k support
points contract it by 0.8^k at lr=0.1. Everything below is checked against
that closed form — including the second-order gradients through the inner
optimization that convergence requires.
"""

import jax.numpy as jnp
import optax

from nodejax import NodeDef, batch, finetune, train_step
from nodejax.struct import Struct
from nodejax.control import Gain
from nodejax.transforms import metasgd as meta_sgd
from nodejax.util import mse, tile


def test_finetune_single_task():
    """finetune() alone: k inner steps from the given param, then query."""
    tuned = finetune(Gain(), mse, optax.sgd(0.1))
    assert isinstance(tuned, NodeDef) and tuned.parametric and not tuned.cyclic

    node = tuned.parameterize(scale=jnp.array(1.0))
    task = Struct(
        support=Struct(input=jnp.ones(3), target=jnp.full(3, 5.0)),
        query=jnp.array(1.0),
    )
    pred = node.apply(task)
    # scale: 1 -> 5 - 0.8^3 * (5 - 1) = 2.952
    assert jnp.allclose(pred, 2.952, atol=1e-5)


def test_maml_convergence():
    """MAML as pure composition: train_step(batch(finetune(model))).

    Two tasks a in {2, 4}. Post-adaptation meta-loss is
    0.8^6 * ((theta-3)^2 + 1) = 0.2621 * ((theta-3)^2 + 1),
    minimized at theta = 3 with value 0.2621 — asserted exactly.
    Converging here requires gradients THROUGH the inner SGD loop
    (second-order), which the purity of the contract form gives for free.
    """
    gain = Gain()
    maml = batch(finetune(gain, mse, optax.sgd(0.1)))
    trainer = train_step(maml, mse, optax.adam(0.1))

    a = jnp.array([2.0, 4.0])
    k = 3
    task_batch = Struct(
        support=Struct(input=jnp.ones((2, k)), target=jnp.tile(a[:, None], (1, k))),
        query=jnp.ones(2),
    )

    state = trainer.init(model=maml.parameterize(scale=jnp.array(0.0)).param)
    steps = 400
    inputs = Struct(input=tile(task_batch, steps), target=tile(a, steps))
    final, losses = trainer.scan(state, inputs)

    # meta-init converges to the analytic optimum: the task mean
    assert jnp.allclose(final.model.scale, 3.0, atol=0.05)
    # and the meta-loss to its analytic floor 0.8^(2k) = 0.2621
    assert jnp.allclose(losses[-1], 0.8 ** (2 * k), atol=0.02)
    # starting loss, for reference: 0.2621 * ((0-3)^2 + 1) = 2.621
    assert jnp.allclose(losses[0], 0.8 ** (2 * k) * 10.0, atol=0.05)


def test_maml_finetunes_to_unseen_task():
    """The meta-learned init finetunes to a task outside the training pair
    better than a naive init does."""
    gain = Gain()
    tuned = finetune(gain, mse, optax.sgd(0.1))
    k = 3

    def query_error(init_scale, a):
        task = Struct(
            support=Struct(input=jnp.ones(k), target=jnp.full(k, a)),
            query=jnp.array(1.0),
        )
        pred = tuned.parameterize(scale=jnp.array(init_scale)).apply(task)
        return jnp.abs(pred - a)

    # meta-optimal init (3.0) vs naive init (0.0) on an unseen task a=3.5
    assert query_error(3.0, 3.5) < query_error(0.0, 3.5)
    # closed form: error = 0.8^k * |init - a|
    assert jnp.allclose(query_error(3.0, 3.5), 0.8 ** k * 0.5, atol=1e-5)


def test_meta_sgd_learns_newton_step():
    """Meta-learning the inner lr. For the quadratic gain task, one inner
    step contracts the error by (1 - 2*lr), so the optimal one-step lr is
    the Newton step 0.5 — at which ANY task is solved in a single inner
    step. The meta-learner should discover it."""
    gain = Gain()
    single = meta_sgd(gain, mse, lr0=0.1)
    trainer = train_step(batch(single), mse, optax.adam(0.05))

    a = jnp.array([2.0, 4.0])
    task_batch = Struct(
        support=Struct(input=jnp.ones((2, 1)), target=a[:, None]),  # ONE support point
        query=jnp.ones(2),
    )
    state = trainer.init(model=single.parameterize(scale=jnp.array(0.0)).param)
    steps = 800
    inputs = Struct(input=tile(task_batch, steps), target=tile(a, steps))
    final, losses = trainer.scan(state, inputs)

    # analytic initial meta-loss: (1 - 2*0.1)^2 * ((0-3)^2 + 1) = 6.4
    assert jnp.allclose(losses[0], 6.4, atol=0.01)
    # the learned lr is the Newton step for this quadratic
    assert jnp.allclose(final.model.lr.scale, 0.5, atol=0.05)
    assert losses[-1] < 1e-2

    # with the learned lr, one inner step nails a task far outside the
    # training pair (a=7 vs training a in {2, 4})
    solo = single.bind(final.model)
    task = Struct(support=Struct(input=jnp.ones(1), target=jnp.full(1, 7.0)),
                  query=jnp.array(1.0))
    assert jnp.abs(solo.apply(task) - 7.0) < 0.5
