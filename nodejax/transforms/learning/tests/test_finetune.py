"""Meta-learning as composition.

finetune() is a transform, so MAML is one line:
train_step(batch(finetune(train_step(model, loss, sgd)))) — learning an init that finetunes well.
finetune extends this to optimizer hyperparameters: meta-learning X means
promoting X from a static capture to a component of param.

Task family: y = a * x, model y = scale * x, inner SGD with lr 0.1 on mse.
One inner step contracts the error (scale - a) by (1 - 2*lr), so k support
points contract it by 0.8^k at lr=0.1. Everything below is checked against
that closed form — including the second-order gradients through the inner
optimization that convergence requires.
"""

import jax.numpy as jnp
import optax

from nodejax.transforms.learning import learned_sgd
from nodejax import trained, scan, PNode, batch, finetune, train_step
from nodejax.struct import Struct
from nodejax.control import Gain
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


from nodejax import Wrapper, serial, nn, map_members, tree_detach

def test_finetune_single_task():
    """finetune() alone: k inner steps from the given param, then query.
    The model's binding rides in through the trainer and comes out
    PROMOTED: the episode node is a PNode on the trainer's own tree."""
    model = Gain().parameterize(scale=jnp.array(1.0)).initialize()
    tuned = finetune(train_step(model, mse, optax.sgd(0.1)))
    assert type(tuned) is PNode and not tuned.cyclic

    task = Struct(
        support=Struct(input=jnp.ones(3), target=jnp.full(3, 5.0)),
        query=jnp.array(1.0),
    )
    pred = tuned.apply(bundle=task)
    # scale: 1 -> 5 - 0.8^3 * (5 - 1) = 2.952
    assert jnp.allclose(pred, 2.952, atol=1e-5)


def test_finetune_resolves_the_optimizer_member():
    """Episode input resolution rebuilds the trainer from both members."""
    import jax

    model = nn.Linear(1).with_input(jnp.zeros(2)).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    tuned = finetune(train_step(model, mse, optax.sgd(0.1)))
    task = Struct(
        support=Struct(input=jnp.ones((3, 2)),
                       target=jnp.full((3, 1), 5.0)),
        query=jnp.ones(2),
    )

    assert tuned.apply(bundle=task).shape == (1,)


def test_finetune_metasgd_ttt_wrapper_rebuild():
    """A rewrite reaches the model through whatever each transform is built
    of, at whatever depth that puts it. finetune is trained(train_step(...))
    now, so its model sits three members down instead of one, and the walk
    does not care: that is the point of declaring members rather than closing
    over them."""

    def member_named(node, key):
        """the node under `key`, wherever in the tree it sits: the three
        transforms are built of different things and put it at different
        depths, and a rewrite is supposed not to care"""
        if key in node.members:
            return getattr(node.members, key)
        for m in node.members:
            found = member_named(m, key)
            if found is not None:
                return found
        return None

    l1 = nn.Linear(4)
    l2 = Wrapper(inner=nn.Linear(4))(name='target')
    import jax
    model = serial(l1=l1, l2=l2).with_input(jnp.zeros(4)).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()

    fn_node = finetune(train_step(model, mse, optax.sgd(0.01)))
    msgd_node = finetune(train_step(model, mse, learned_sgd(0.01)))
    ttt_node = train_step(model, mse, learned_sgd(0.01)).pnode

    def negate(self, input):
        return -self.body(input)

    new_l2 = Wrapper(body=l2)(negate, name='replacement')
    for transform_node in (fn_node, msgd_node, ttt_node):
        rebuilt = map_members(
            transform_node.node,
            lambda m: new_l2 if m.name == 'target' else m)
        assert member_named(rebuilt, 'l2').name == 'replacement'

        detached = tree_detach(transform_node, 'l2')
        assert member_named(detached.node, 'l2').name == 'detach(target)'

    assert l2.name == 'target'

    task = Struct(
        support=Struct(
            input=jnp.ones((1, 4)), target=jnp.zeros((1, 4))),
        query=jnp.ones(4),
    )
    rebuilt_node = map_members(
        fn_node.node, lambda m: new_l2 if m.name == 'target' else m)
    rebuilt = rebuilt_node.bind(fn_node.param)
    assert jnp.allclose(
        rebuilt.apply(bundle=task), -fn_node.apply(bundle=task))



def test_maml_convergence():
    """MAML as pure composition: train_step(batch(finetune(train_step(model, loss, sgd)))).

    Two tasks a in {2, 4}. Post-adaptation meta-loss is
    0.8^6 * ((theta-3)^2 + 1) = 0.2621 * ((theta-3)^2 + 1),
    minimized at theta = 3 with value 0.2621 — asserted exactly.
    Converging here requires gradients THROUGH the inner SGD loop
    (second-order), which the purity of the contract form gives for free.
    """
    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    maml = batch(finetune(train_step(model, mse, optax.sgd(0.1))))
    trainer = train_step(maml.initialize(), mse, optax.adam(0.1))

    a = jnp.array([2.0, 4.0])
    k = 3
    task_batch = Struct(
        support=Struct(input=jnp.ones((2, k)), target=jnp.tile(a[:, None], (1, k))),
        query=jnp.ones(2),
    )

    steps = 400
    final, (_, aux) = trainer.scan(
        support=tile(task_batch.support, steps),
        query=tile(task_batch.query, steps),
        target=tile(a, steps),
    )

    # meta-init converges to the analytic optimum: the task mean
    assert jnp.allclose(
        final.state.opt.params.model.objective.model.scale,
        3.0,
        atol=0.05,
    )
    # and the meta-loss to its analytic floor 0.8^(2k) = 0.2621
    assert jnp.allclose(aux.loss[-1], 0.8 ** (2 * k), atol=0.02)
    # starting loss, for reference: 0.2621 * ((0-3)^2 + 1) = 2.621
    assert jnp.allclose(aux.loss[0], 0.8 ** (2 * k) * 10.0, atol=0.05)


def test_maml_finetunes_to_unseen_task():
    """The meta-learned init finetunes to a task outside the training pair
    better than a naive init does."""
    k = 3

    def query_error(init_scale, a):
        model = Gain().parameterize(scale=jnp.array(init_scale)).initialize()
        tuned = finetune(train_step(model, mse, optax.sgd(0.1)))
        task = Struct(
            support=Struct(input=jnp.ones(k), target=jnp.full(k, a)),
            query=jnp.array(1.0),
        )
        return jnp.abs(tuned.apply(bundle=task) - a)

    # meta-optimal init (3.0) vs naive init (0.0) on an unseen task a=3.5
    assert query_error(3.0, 3.5) < query_error(0.0, 3.5)
    # closed form: error = 0.8^k * |init - a|
    assert jnp.allclose(query_error(3.0, 3.5), 0.8 ** k * 0.5, atol=1e-5)


def test_meta_sgd_learns_newton_step():
    """Meta-learning the inner lr. For the quadratic gain task, one inner
    step contracts the error by (1 - 2*lr), so the optimal one-step lr is
    the Newton step 0.5 — at which ANY task is solved in a single inner
    step. The meta-learner should discover it."""
    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    single = finetune(train_step(model, mse, learned_sgd(0.1)))
    trainer = train_step(batch(single).initialize(), mse, optax.adam(0.05))

    a = jnp.array([2.0, 4.0])
    task_batch = Struct(
        support=Struct(input=jnp.ones((2, 1)), target=a[:, None]),  # ONE support point
        query=jnp.ones(2),
    )
    steps = 800
    final, (_, aux) = trainer.scan(
        support=tile(task_batch.support, steps),
        query=tile(task_batch.query, steps),
        target=tile(a, steps),
    )

    # analytic initial meta-loss: (1 - 2*0.1)^2 * ((0-3)^2 + 1) = 6.4
    assert jnp.allclose(aux.loss[0], 6.4, atol=0.01)
    # the learned lr is the Newton step for this quadratic
    assert jnp.allclose(
        final.state.opt.params.model.opt.model.scale,
        0.5,
        atol=0.05,
    )
    assert aux.loss[-1] < 1e-2

    # with the learned lr, one inner step nails a task far outside the
    # training pair (a=7 vs training a in {2, 4})
    solo = single.node.bind(final.state.opt.params.model)
    task = Struct(support=Struct(input=jnp.ones(1), target=jnp.full(1, 7.0)),
                  query=jnp.array(1.0))
    assert jnp.abs(solo.apply(bundle=task) - 7.0) < 0.5
