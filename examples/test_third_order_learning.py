"""Third-order differentiation through three nested `train_step` Nodes.

`supervised_ttt` adapts `Scale` online within a sequence, `finetune`
differentiates through that online adaptation over support episodes, and an
outer `train_step` differentiates through episodic fine-tuning on the query. All
three levels apply the same `train_step` transform without level-specific
differentiation hooks.

Using the quartic test-time loss `weight**4 / 4` produces a nonzero third
derivative. `test_outer_gradient_contains_the_inner_third_derivative` checks the
observed outer gradient against the exact analytical expression and verifies
that dropping the third-derivative term breaks agreement. Executing
`python -m examples.test_third_order_learning` prints the numerical breakdown.
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
    supervised_ttt,
    train_step,
)
from nodejax.core.types import PyTree
from examples import third_order_learning as showcase


TTT_RATE = 0.25
FINETUNE_RATE = 0.5
META_RATE = 0.01


def Scale() -> Node:
    """Multiplies its input by a scalar trainable weight initialized to `1.0`."""
    def param(weight: PyTree = 1.0) -> PyTree:
        return jnp.asarray(weight)

    def apply(param, input):
        return param * input

    return Leaf(apply, param=param, name='scale')


def quartic(output: jax.Array, target: jax.Array) -> jax.Array:
    """Computes one-quarter of the fourth power of prediction error."""
    error = output - target
    return 0.25 * error ** 4


def terminal_quadratic(output: PyTree, terminal_target: jax.Array) -> jax.Array:
    """Computes half the squared error between the final sequence prediction and `terminal_target`."""
    return 0.5 * (output[-1] - terminal_target) ** 2


def third_order_learner() -> Node:
    """Composes online test-time training, episodic fine-tuning, and meta-training from three `train_step` Nodes."""
    online = scanned(supervised_ttt(
        train_step(Scale(), quartic, optax.sgd(TTT_RATE))))
    adapted = finetune(
        train_step(online, terminal_quadratic, optax.sgd(FINETUNE_RATE)))
    return train_step(adapted, terminal_quadratic, optax.sgd(META_RATE))


def task() -> Struct:
    """Constructs a deterministic two-step scalar episode for closed-form gradient verification."""
    sequence = Struct(input=jnp.ones(2), target=jnp.zeros(2))
    support = Struct(
        input=jnp.ones((1, 2)),
        target=jnp.zeros((1, 2)),
        terminal_target=jnp.zeros(1),
    )
    return Struct(support=support, query=sequence)


def run() -> Struct:
    """Evaluates one outer `train_step` update and compares the observed gradient against closed-form formulas."""
    trainer = third_order_learner().parameterize().initialize()
    episode = task()
    successor, (_, aux) = jax.jit(trainer.apply)(
        support=episode.support,
        query=episode.query,
        terminal_target=jnp.asarray(0.0),
    )

    initial_weight = trainer.state.opt.params.model.objective.model.objective.model
    updated_weight = successor.state.opt.params.model.objective.model.objective.model
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
    """Verifies that the outer meta-gradient matches the exact third-order formula and differs when the third derivative is omitted."""
    result = run()
    assert jnp.allclose(result.observed, result.exact, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(result.observed, result.without_third, rtol=1e-3, atol=1e-3)


def test_full_showcase_trains_through_the_tied_recurrent_tree() -> None:
    """Verifies `BatchNorm` state axes, parameter updates, parameter tying, and tapped `Aux` shapes on `third_order_learning`."""
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
    assert result.aux.loss.shape == (showcase.META_STEPS,)
    assert bool(jnp.all(jnp.isfinite(result.aux.loss)))
    assert not jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            result.final.param,
            result.initial.param.objective.model,
        )
    )

    paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(result.final.param)[0]
    }
    assert any('.encoder.weight' in path for path in paths)
    assert not any('.decoder.weight' in path for path in paths)

    tapped = [
        leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(result.aux)[0]
        if jax.tree_util.keystr(path).endswith('.tanh')
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
    """Prints the outer loss, observed meta-gradient, and analytical third-derivative contribution."""
    result = run()
    contribution = result.exact - result.without_third
    print(f'outer loss: {float(result.loss):.7f}')
    print(f'NodeJAX outer gradient: {float(result.observed):.7f}')
    print(f'exact third-order gradient: {float(result.exact):.7f}')
    print(f'without the third derivative: {float(result.without_third):.7f}')
    print(f'third-derivative contribution: {float(contribution):.7f}')


if __name__ == '__main__':
    main()
