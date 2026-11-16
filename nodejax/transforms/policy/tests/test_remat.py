"""Verifies activation checkpointing via `remat` across stacked and scanned Nodes.

`remat` preserves outputs, gradients, parameter structure, and state structure
while inserting `jax.checkpoint` into the backward jaxpr at the wrapped boundary.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, nn, batch, scan, scanned, stack, remat, train_step, tree_freeze, tree_filter
from nodejax.struct import Struct

X = jnp.ones((4, 8))


def _layer() -> Node:
    return (nn.Linear(8) >> nn.gelu).node


def _tower(layer: Node, n: int=4):
    return batch(stack(layer, n=n)).with_input(X).parameterize(rng=jax.random.PRNGKey(0))


def _loss(model) -> jax.Array:
    return jnp.sum(model.apply(X) ** 2)


def test_remat_changes_nothing_the_caller_can_see():
    """Verifies that `remat` preserves forward outputs, parameter gradients, and parameter PyTree structure."""
    plain, remade = _tower(_layer()), _tower(remat(_layer()))

    assert jnp.allclose(plain.apply(X), remade.apply(X))

    g_plain = jax.tree.leaves(jax.grad(_loss)(plain))
    g_remat = jax.tree.leaves(jax.grad(_loss)(remade))
    assert len(g_plain) == len(g_remat)
    for a, b in zip(g_plain, g_remat):
        assert jnp.allclose(a, b, atol=1e-5)

    assert jax.tree.structure(plain.param) == jax.tree.structure(remade.param)


def test_the_recomputation_is_actually_there():
    """Verifies that `remat` emits the `remat` primitive in the backward jaxpr."""
    plain = str(jax.make_jaxpr(jax.grad(_loss))(_tower(_layer())))
    remade = str(jax.make_jaxpr(jax.grad(_loss))(_tower(remat(_layer()))))

    assert 'remat' not in plain
    assert 'remat' in remade


def test_remat_composes_where_depth_is_spelled():
    """Verifies that `remat` composes inside `batch(stack(...))` under `train_step`."""
    layer = remat(_layer())
    model = _tower(layer, n=6).initialize()
    trainer = train_step(
        model,
        lambda prediction, target: jnp.mean((prediction - target) ** 2),
        optax.adam(1e-2),
    )
    steps = 20
    final, (_, aux) = trainer.scan(input=jnp.broadcast_to(X, (steps, *X.shape)),
                                   target=jnp.zeros((steps, 4, 8)))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]


def test_remat_is_a_wrapper_like_any_other():
    """Verifies that `tree_freeze` traverses through `remat` to freeze cyclic inner members."""
    pipe = (nn.Linear(4) >> nn.EMA(0.1)).node
    model = batch(remat(pipe)).with_input(jnp.ones((3, 4))).parameterize(
        rng=jax.random.PRNGKey(0))

    state = model.init(input=jnp.ones((3, 4)))
    frozen = tree_freeze(model, tree_filter(state, 'ema'))
    assert not frozen.cyclic


def test_where_it_sits_is_the_granularity():
    """Verifies that `scanned(remat(cell))` and `remat(scanned(cell))` both match exact uncheckpointed gradients."""
    seq = jnp.ones((16, 4))

    def grads(node):
        model = node.with_input(seq).parameterize(rng=jax.random.PRNGKey(0))
        loss = lambda m: jnp.sum(m.apply(seq) ** 2)
        return (jax.tree.leaves(jax.grad(loss)(model)),
                'remat' in str(jax.make_jaxpr(jax.grad(loss))(model)))

    plain, plain_has = grads(scanned(nn.RNN(4)))
    per_step, step_has = grads(scanned(remat(nn.RNN(4))))
    rollout, roll_has = grads(remat(scanned(nn.RNN(4))))

    assert not plain_has and step_has and roll_has
    for a, b, c in zip(plain, per_step, rollout):
        assert jnp.allclose(a, b, atol=1e-4)
        assert jnp.allclose(a, c, atol=1e-4)


def test_a_substituted_member_reaches_the_apply():
    """Verifies that `remat` reads its wrapped member from `contract.members` when `map_members` substitutes it."""
    from nodejax import Leaf, map_members

    def Scale(k):
        return Leaf(lambda input: input * k, name='scale').node

    wrapped = remat(Scale(2.0)).node
    swapped = map_members(
        wrapped,
        lambda member: Scale(10.0)
        if member.name == 'scale' else member,
    )
    assert jnp.allclose(
        swapped.parameterize().apply(jnp.asarray(1.0)), 10.0)
