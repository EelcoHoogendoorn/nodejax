"""Transfer learning: pretrain, freeze the trunk, swap the head, retrain.

The whole workflow is param surgery plus one transform: params are data,
so 'swap the head' is building a param tree from the trained trunk and a
fresh head; 'freeze the trunk' is tree_detach, which stops gradient
through the matched members while everything else trains. The frozen
trunk is bitwise untouched by the second training run.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.types import PyTree
from nodejax.struct import Struct
from nodejax import Node, node, trained, Leaf, serial, tree_detach, train_step
from nodejax import tile
from nodejax.examples.util import mse
N_IN, FEAT = 4, 16


@node
def Trunk() -> Node:
    def param(rng):
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (N_IN, FEAT)),
                      w2=0.5 * jax.random.normal(rng.next(), (FEAT, FEAT)))

    def apply(param, input):
        return jnp.tanh(jnp.tanh(input @ param.w1) @ param.w2)

    return Leaf(apply, param=param)


@node
def Head() -> Node:
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (FEAT, 1)))

    def apply(param, input):
        return input @ param.w

    return Leaf(apply, param=param)


def _fit(node, params: PyTree, X, y: jax.Array, steps: int, lr: float=1e-2):
    trainer = train_step(node.bind(params).initialize(), mse, optax.adam(lr))
    done, aux = trained(trainer).apply(input=tile(X, steps), target=tile(y, steps))
    return done.param, aux.loss


def test_freeze_trunk_swap_head_retrain():
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y_pretrain = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    pipe = serial(trunk=Trunk(), head=Head())

    # task A: pretrain the whole pipe
    init = pipe.parameterize(rng=jax.random.PRNGKey(1)).param
    pretrained, losses_a = _fit(pipe, init, X, y_pretrain, steps=500)
    assert losses_a[-1] < 0.05

    # task B: targets linear in the LEARNED features, so a fresh head can
    # fit them exactly iff the trunk survives intact
    feats = Trunk().apply(pretrained.trunk, X)
    y_transfer = feats @ jax.random.normal(jax.random.PRNGKey(2), (FEAT, 1))

    # swap the head (params are data), freeze the trunk (one transform)
    fresh_head = Head().parameterize(rng=jax.random.PRNGKey(3)).param
    surgery = pretrained.replace(head=fresh_head)
    tuned, losses_b = _fit(tree_detach(pipe, 'trunk'), surgery, X, y_transfer,
                           steps=1500, lr=3e-2)

    assert losses_b[-1] < 1e-3 * jnp.mean(y_transfer ** 2)   # head learned task B
    assert jax.tree.all(jax.tree.map(              # trunk bitwise untouched
        jnp.array_equal, tuned.trunk, pretrained.trunk))
    assert not jax.tree.all(jax.tree.map(          # head actually trained
        jnp.array_equal, tuned.head, fresh_head))
