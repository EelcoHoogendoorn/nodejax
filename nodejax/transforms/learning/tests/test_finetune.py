"""Verifies `finetune` episodes, MAML outer optimization, and Meta-SGD step sizes."""

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
    """Verifies `finetune` runs support steps from the bound parameter and evaluates the query."""
    model = Gain().parameterize(scale=jnp.array(1.0)).initialize()
    tuned = finetune(train_step(model, mse, optax.sgd(0.1)))
    assert type(tuned) is PNode and not tuned.cyclic

    task = Struct(
        support=Struct(input=jnp.ones(3), target=jnp.full(3, 5.0)),
        query=jnp.array(1.0),
    )
    pred = tuned.apply(bundle=task)
    # Closed-form scale after 3 support steps: 5 - 0.8^3 * (5 - 1) = 2.952.
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
    """Verifies `map_members` and `tree_detach` reach the inner model across `finetune`, Meta-SGD, and TTT trees."""

    def member_named(node, key):
        """Finds the member registered under `key` at any depth in `node`."""
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
    """Verifies MAML outer optimization converges to the closed-form meta-initialization and loss floor."""
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

    # Meta-initialization converges to the task mean 3.0.
    assert jnp.allclose(
        final.state.opt.params.model.objective.model.scale,
        3.0,
        atol=0.05,
    )
    # Meta-loss converges to 0.8^(2k) = 0.2621.
    assert jnp.allclose(aux.loss[-1], 0.8 ** (2 * k), atol=0.02)
    # Initial meta-loss equals 0.2621 * ((0 - 3)^2 + 1) = 2.621.
    assert jnp.allclose(aux.loss[0], 0.8 ** (2 * k) * 10.0, atol=0.05)


def test_maml_finetunes_to_unseen_task():
    """Verifies the meta-learned initialization adapts to an unseen task with lower query error than zero initialization."""
    k = 3

    def query_error(init_scale, a):
        model = Gain().parameterize(scale=jnp.array(init_scale)).initialize()
        tuned = finetune(train_step(model, mse, optax.sgd(0.1)))
        task = Struct(
            support=Struct(input=jnp.ones(k), target=jnp.full(k, a)),
            query=jnp.array(1.0),
        )
        return jnp.abs(tuned.apply(bundle=task) - a)

    # Compares the meta-optimal initialization (3.0) against zero initialization on unseen a = 3.5.
    assert query_error(3.0, 3.5) < query_error(0.0, 3.5)
    # Closed-form query error equals 0.8^k * |init - a|.
    assert jnp.allclose(query_error(3.0, 3.5), 0.8 ** k * 0.5, atol=1e-5)


def test_meta_sgd_learns_newton_step():
    """Verifies outer optimization over `learned_sgd` converges to the quadratic Newton step size `0.5`."""
    model = Gain().parameterize(scale=jnp.array(0.0)).initialize()
    single = finetune(train_step(model, mse, learned_sgd(0.1)))
    trainer = train_step(batch(single).initialize(), mse, optax.adam(0.05))

    a = jnp.array([2.0, 4.0])
    task_batch = Struct(
        support=Struct(input=jnp.ones((2, 1)), target=a[:, None]),
        query=jnp.ones(2),
    )
    steps = 800
    final, (_, aux) = trainer.scan(
        support=tile(task_batch.support, steps),
        query=tile(task_batch.query, steps),
        target=tile(a, steps),
    )

    # Analytic initial meta-loss equals (1 - 2 * 0.1)^2 * ((0 - 3)^2 + 1) = 6.4.
    assert jnp.allclose(aux.loss[0], 6.4, atol=0.01)
    # Learned step size reaches the Newton step 0.5.
    assert jnp.allclose(
        final.state.opt.params.model.opt.model.scale,
        0.5,
        atol=0.05,
    )
    assert aux.loss[-1] < 1e-2

    # One inner step with the learned rate adapts to unseen task a = 7.
    solo = single.node.bind(final.state.opt.params.model)
    task = Struct(support=Struct(input=jnp.ones(1), target=jnp.full(1, 7.0)),
                  query=jnp.array(1.0))
    assert jnp.abs(solo.apply(bundle=task) - 7.0) < 0.5
