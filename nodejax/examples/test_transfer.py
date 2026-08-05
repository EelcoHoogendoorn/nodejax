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

from nodejax.struct import Struct
from nodejax import node_def, serial, tree_detach, train_step
from nodejax.examples import mse, tile

N_IN, FEAT = 4, 16


def trunk_def():
    def param(rng):
        return Struct(w1=0.5 * jax.random.normal(rng.next(), (N_IN, FEAT)),
                      w2=0.5 * jax.random.normal(rng.next(), (FEAT, FEAT)))

    def apply(param, input):
        return jnp.tanh(jnp.tanh(input @ param.w1) @ param.w2)

    return node_def(apply, param=param, name='trunk')


def head_def():
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (FEAT, 1)))

    def apply(param, input):
        return input @ param.w

    return node_def(apply, param=param, name='head')


def _fit(node, params, X, y, steps, lr=1e-2):
    trainer = train_step(node, mse, optax.adam(lr))
    state = trainer.init(model=params)
    state, losses = trainer.scan(state, Struct(input=tile(X, steps), target=tile(y, steps)))
    return state.model, losses


def test_freeze_trunk_swap_head_retrain():
    X = jax.random.normal(jax.random.PRNGKey(0), (64, N_IN))
    y_pretrain = jnp.sin(X @ jnp.array([[1.0], [-1.0], [0.5], [0.0]]))

    pipe = serial(trunk=trunk_def(), head=head_def())

    # task A: pretrain the whole pipe
    init = pipe.parameterize(rng=jax.random.PRNGKey(1)).param
    pretrained, losses_a = _fit(pipe, init, X, y_pretrain, steps=500)
    assert losses_a[-1] < 0.05

    # task B: targets linear in the LEARNED features, so a fresh head can
    # fit them exactly iff the trunk survives intact
    feats = trunk_def().apply(pretrained.trunk, X)
    y_transfer = feats @ jax.random.normal(jax.random.PRNGKey(2), (FEAT, 1))

    # swap the head (params are data), freeze the trunk (one transform)
    fresh_head = head_def().parameterize(rng=jax.random.PRNGKey(3)).param
    surgery = pretrained.replace(head=fresh_head)
    tuned, losses_b = _fit(tree_detach(pipe, 'trunk'), surgery, X, y_transfer,
                           steps=1500, lr=3e-2)

    assert losses_b[-1] < 1e-3 * jnp.mean(y_transfer ** 2)   # head learned task B
    assert jax.tree.all(jax.tree.map(              # trunk bitwise untouched
        jnp.array_equal, tuned.trunk, pretrained.trunk))
    assert not jax.tree.all(jax.tree.map(          # head actually trained
        jnp.array_equal, tuned.head, fresh_head))
