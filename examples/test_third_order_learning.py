"""Third-order learning from ordinary Node composition.

The innermost train step adapts a model during a sequence. Fine-tuning differentiates through that
adaptation over support data. The outer train step differentiates through fine-tuning on the query.
Nothing in any of the three train steps knows which level it serves.

The scalar task keeps the derivative visible. Its TTT loss is ``weight**4 / 4``, whose third
derivative is nonzero. The test compares the gradient produced by the composed Nodes with its exact
formula, then removes only that third derivative from the formula and shows that the result changes.

Run directly: ``python -m examples.test_third_order_learning``
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Leaf,
    Node,
    Struct,
    batch,
    finetune,
    scanned,
    split_aux,
    supervised_ttt,
    train_step,
)
from nodejax.core.types import PyTree
from examples import third_order_learning as showcase


TTT_RATE = 0.25
FINETUNE_RATE = 0.5
META_RATE = 0.01


def Scale() -> Node:
    def param(weight: PyTree = 1.0) -> PyTree:
        return jnp.asarray(weight)

    def apply(param, input):
        return param * input

    return Leaf(apply, param=param, name='scale')


def quartic(output: jax.Array, target: jax.Array) -> jax.Array:
    error = output - target
    return 0.25 * error ** 4


def terminal_quadratic(output: PyTree, target: jax.Array) -> jax.Array:
    prediction, _ = split_aux(output)
    return 0.5 * (prediction[-1] - target) ** 2


def third_order_learner() -> Node:
    online = scanned(supervised_ttt(
        train_step(Scale(), quartic, optax.sgd(TTT_RATE))))
    adapted = finetune(
        train_step(online, terminal_quadratic, optax.sgd(FINETUNE_RATE)))
    return train_step(adapted, terminal_quadratic, optax.sgd(META_RATE))


def task() -> Struct:
    sequence = Struct(input=jnp.ones(2), target=jnp.zeros(2))
    support = Struct(
        input=Struct(input=jnp.ones((1, 2)), target=jnp.zeros((1, 2))),
        target=jnp.zeros(1),
    )
    return Struct(support=support, query=sequence)


def run() -> Struct:
    trainer = third_order_learner().parameterize(weight=1.0).initialize()
    successor, (_, aux) = jax.jit(trainer.apply)(input=task(), target=jnp.asarray(0.0))

    initial_weight = trainer.state.opt.params.model.model
    updated_weight = successor.state.opt.params.model.model
    observed = (initial_weight - updated_weight) / META_RATE

    weight = jnp.asarray(1.0)
    fast_weight = weight - TTT_RATE * weight ** 3
    fast_slope = 1 - 3 * TTT_RATE * weight ** 2
    adapted_weight = weight - FINETUNE_RATE * fast_weight * fast_slope
    query_weight = adapted_weight - TTT_RATE * adapted_weight ** 3
    query_slope = 1 - 3 * TTT_RATE * adapted_weight ** 2
    inner_third = 6 * weight

    exact = query_weight * query_slope * (
        1 - FINETUNE_RATE * (
            fast_slope ** 2 - TTT_RATE * fast_weight * inner_third
        )
    )
    without_third = query_weight * query_slope * (
        1 - FINETUNE_RATE * fast_slope ** 2
    )
    return Struct(
        loss=aux.loss,
        observed=observed,
        exact=exact,
        without_third=without_third,
    )


def test_outer_gradient_contains_the_inner_third_derivative() -> None:
    result = run()
    assert jnp.allclose(result.observed, result.exact, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(result.observed, result.without_third, rtol=1e-3, atol=1e-3)


def test_full_showcase_trains_through_the_tied_recurrent_tree() -> None:
    input = jnp.zeros((showcase.TASKS, showcase.FEATURES))
    batched_predictor = batch(showcase.predictor()).with_input(input).parameterize(
        rng=jax.random.PRNGKey(0)
    ).initialize(input=input)
    running = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(batched_predictor.state)[0]
        if '.model.batch_norm.' in jax.tree_util.keystr(path)
    }
    assert next(value for path, value in running.items() if path.endswith('.mean')).shape == (
        showcase.DEPTH,
        showcase.WIDTH,
    )
    assert next(value for path, value in running.items() if path.endswith('.var')).shape == (
        showcase.DEPTH,
        showcase.WIDTH,
    )

    result = showcase.run()
    assert bool(jnp.all(jnp.isfinite(result.aux.loss)))
    quarter = showcase.META_STEPS // 4
    assert jnp.mean(result.aux.loss[-quarter:]) < 0.5 * result.aux.loss[0]

    paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(result.final.param)[0]
    }
    assert any('.encoder.weight' in path for path in paths)
    assert not any('.decoder.weight' in path for path in paths)

    tapped = [
        leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(result.aux)[0]
        if jax.tree_util.keystr(path).endswith('.recurrent.activation')
    ]
    assert len(tapped) == 1
    assert tapped[0].shape == (
        showcase.META_STEPS,
        showcase.TASKS,
        showcase.MEMBERS,
        showcase.STREAM,
        showcase.WIDTH,
    )


def main() -> None:
    result = run()
    contribution = result.exact - result.without_third
    print(f'outer loss: {float(result.loss):.7f}')
    print(f'NodeJAX outer gradient: {float(result.observed):.7f}')
    print(f'exact third-order gradient: {float(result.exact):.7f}')
    print(f'without the third derivative: {float(result.without_third):.7f}')
    print(f'third-derivative contribution: {float(contribution):.7f}')


if __name__ == '__main__':
    main()
