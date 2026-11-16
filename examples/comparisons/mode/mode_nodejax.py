"""Train and evaluation mode construction in NodeJAX.

NodeJAX represents evaluation as a distinct `PNode` rather than a mutable flag or call-time boolean. `nn.eval_mode(final)` runs `final.specialize(**{'*.train': False})` so `nn.Dropout` rebuilds as the identity with no `rng` requirement, and freezes `tag='running_stats'` state slots via `tree_freeze` so `nn.BatchNorm` folds its accumulated statistics into constants (`not evaluator.cyclic`).

Run directly with `python -m examples.comparisons.mode.mode_nodejax`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import Node, Struct, batch, nn, tile, train_step, trained, tree_freeze

from examples.comparisons.mode.mode_common import (
    CLASSES, HIDDEN, LR, MOMENTUM, RATE, STEPS, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


def build(drop_rate: float) -> Node:
    return (nn.Linear(HIDDEN) >> nn.gelu >> nn.Dropout(drop_rate)
            >> nn.BatchNorm(MOMENTUM) >> nn.Linear(CLASSES))


def main() -> None:
    xs_np, ys_np = make_data(np.random.RandomState(0))
    xs, ys = jnp.asarray(xs_np), jnp.asarray(ys_np)

    trainer = train_step(batch(build(RATE)), xent, optax.adam(LR))
    trainer = trainer.with_input(Struct(input=xs, target=ys)).parameterize(
        rng=jax.random.PRNGKey(1)
    )
    final, aux = trained(trainer).apply(
        rng=jax.random.PRNGKey(0), input=tile(xs, STEPS), target=tile(ys, STEPS)
    )

    _, logits_a = final(input=xs, rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=xs, rng=jax.random.PRNGKey(3))
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    # nn.eval_mode(final) packages specialize(**{'*.train': False}) + tree_freeze
    evaluator = nn.eval_mode(final)
    manual = tree_freeze(
        final.specialize(**{'*.train': False}).bind(final.param, state=final.state),
        tag='running_stats',
    )
    assert not evaluator.cyclic and evaluator.state == ()
    _, eval_a = evaluator(xs)
    _, manual_a = manual(xs)
    assert jnp.allclose(eval_a, manual_a)

    _, eval_b = evaluator(xs)
    _, solo = evaluator(xs[:16])
    report(
        'nodejax',
        train_stochastic,
        bool(jnp.allclose(eval_a, eval_b)),
        bool(jnp.allclose(solo, eval_a[:16], atol=1e-6)),
        float(aux.loss[0]),
        float(aux.loss[-1]),
    )


if __name__ == '__main__':
    main()
