"""The mode switch, the nodejax side of the comparison.

THERE IS NO MODE. Dropout's rate is a STATIC: eval is the rate-0 build,
which is the identity with no state, so a model whose only stochastic
member is dropout comes out non-cyclic. The batch norm's running stats
are ordinary cyclic state: eval freezes them at what training left
them. The evaluator is therefore a DIFFERENT, SMALLER architecture with
the SAME params bound: nothing stochastic and nothing accumulating
remains, so it is a plain function of its input, and the checks below
hold by construction rather than by flag discipline.

THE MECHANISM is specialize: statics are first-class metadata on the
def, so the eval build comes from the trained model itself,
fitted.specialize(**{'*.train': False}), with no binding kept; every
mode-aware block flipping at once. The stats freeze selects by the
running_stats TAG, what the state IS, never by what a layer happens
to be called.

Run directly:  python -m examples.comparisons.mode.mode_nodejax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import Node, nn, batch, train_step, trained, tree_freeze
from nodejax.struct import Struct

from examples.comparisons.mode.mode_common import (
    HIDDEN, CLASSES, STEPS, RATE, MOMENTUM, LR, make_data, xent, report)


def build(drop_rate: float) -> Node:
    """The architecture, the drop rate a construction static. The train
    and eval models are two builds of this one definition."""
    return (nn.Linear(HIDDEN) >> nn.gelu >> nn.Dropout(drop_rate)
            >> nn.BatchNorm(MOMENTUM) >> nn.Linear(CLASSES))


def main() -> None:
    xs, ys = make_data(np.random.RandomState(0))
    trainer = train_step(batch(build(RATE)), xent, optax.adam(LR))
    trainer = trainer.with_input(Struct(input=xs, target=ys)).parameterize(
        rng=jax.random.PRNGKey(1))
    tile = lambda leaf: jnp.broadcast_to(leaf, (STEPS, *leaf.shape))
    final, aux = trained(trainer).apply(
        rng=jax.random.PRNGKey(0), input=tile(xs), target=tile(ys))

    # train mode is stochastic: dropout draws at apply, so the train
    # build owes a key per call and different keys draw different masks
    # (`final` IS the trained model, state-bound; the batchnorm stats it
    # carries ride along unchanged)
    _, logits_a = final(input=xs, rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=xs, rng=jax.random.PRNGKey(3))
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    # THE MODE SWITCH, on one object: specialize re-binds the statics on
    # the node training used (the dropout leaf rebuilds as the identity,
    # its slot pruned from the carried state; the same params ride), and
    # tree_freeze pins the norm stats it holds. Non-cyclic, and provably
    # so: the evaluator's successor equals it, and discarding is safe
    eval_node = final.specialize(**{'*.train': False})
    evaluator = tree_freeze(
        eval_node.bind(final.param, state=final.state),
        tag='running_stats')
    assert not evaluator.cyclic and evaluator.state == ()

    _, eval_a = evaluator(xs)
    _, eval_b = evaluator(xs)
    _, solo = evaluator(xs[:16])
    report('nodejax',
           train_stochastic,
           bool(jnp.allclose(eval_a, eval_b)),
           bool(jnp.allclose(solo, eval_a[:16], atol=1e-6)),
           float(aux.loss[0]), float(aux.loss[-1]))


if __name__ == '__main__':
    main()
