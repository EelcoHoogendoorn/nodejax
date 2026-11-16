"""Transfer learning via parameter tree replacement and `tree_detach`.

Pretraining optimizes `serial(trunk=Trunk(), head=Head())` end to end on a source
task. Replacing `pretrained.replace(head=fresh_head)` swaps in a newly
initialized readout on the parameter `Struct`, and `tree_detach(pipe, 'trunk')`
stops gradients into `trunk` during target-task training so trunk weights remain
bitwise identical while `head` adapts.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.core.types import PyTree
from nodejax.struct import Struct
from nodejax import Node, node, trained, Leaf, serial, tree_detach, train_step
from nodejax import tile
from examples.util import mse
N_IN, FEAT = 4, 16


@node
def Trunk() -> Node:
    """Computes `FEAT`-dimensional features through two `tanh` linear layers."""
    def param(rng):
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (N_IN, FEAT)),
                      w2=0.5 * jax.random.normal(rng.next(), (FEAT, FEAT)))

    def apply(param, input):
        return jnp.tanh(jnp.tanh(input @ param.w1) @ param.w2)

    return Leaf(apply, param=param)


@node
def Head() -> Node:
    """Projects `FEAT`-dimensional features to a scalar prediction without bias."""
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (FEAT, 1)))

    def apply(param, input):
        return input @ param.w

    return Leaf(apply, param=param)


def _fit(
    node: Node, params: PyTree, X: jax.Array, y: jax.Array, steps: int,
    lr: float = 1e-2,
) -> tuple[PyTree, jax.Array]:
    trainer = train_step(node.bind(params).initialize(), mse, optax.adam(lr))
    done, aux = trained(trainer).apply(input=tile(X, steps), target=tile(y, steps))
    return done.param, aux.loss


def test_freeze_trunk_swap_head_retrain() -> None:
    """Verifies that `tree_detach` leaves pretrained `trunk` weights bitwise unchanged while fitting a replacement `head`."""
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y_pretrain = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    pipe = serial(trunk=Trunk(), head=Head())

    init = pipe.parameterize(rng=jax.random.PRNGKey(1)).param
    pretrained, losses_a = _fit(pipe, init, X, y_pretrain, steps=500)
    assert losses_a[-1] < 0.05

    # Target labels are a linear projection of the pretrained trunk features.
    feats = Trunk().apply(pretrained.trunk, X)
    y_transfer = feats @ jax.random.normal(jax.random.PRNGKey(2), (FEAT, 1))

    # Replace `head` in the parameter `Struct` and detach `trunk` gradients during fine-tuning.
    fresh_head = Head().parameterize(rng=jax.random.PRNGKey(3)).param
    surgery = pretrained.replace(head=fresh_head)
    tuned, losses_b = _fit(tree_detach(pipe, 'trunk'), surgery, X, y_transfer,
                           steps=1500, lr=3e-2)

    assert losses_b[-1] < 1e-3 * jnp.mean(y_transfer ** 2)
    assert jax.tree.all(jax.tree.map(
        jnp.array_equal, tuned.trunk, pretrained.trunk))
    assert not jax.tree.all(jax.tree.map(
        jnp.array_equal, tuned.head, fresh_head))
