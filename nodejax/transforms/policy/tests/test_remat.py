"""remat: invisible to the caller, visible in the jaxpr.

The claim a rematerialization transform has to make is that it changes
nothing a caller can observe — outputs, gradients, params, state — while
changing what the backward pass keeps. The first half is asserted
directly; the second is read off the jaxpr, since a wrapper that silently
did nothing would pass every value check.
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
    """Same output and same gradients, to the last bit the check allows."""
    plain, remade = _tower(_layer()), _tower(remat(_layer()))

    assert jnp.allclose(plain.apply(X), remade.apply(X))

    g_plain = jax.tree.leaves(jax.grad(_loss)(plain))
    g_remat = jax.tree.leaves(jax.grad(_loss)(remade))
    assert len(g_plain) == len(g_remat)
    for a, b in zip(g_plain, g_remat):
        assert jnp.allclose(a, b, atol=1e-5)

    # and the param tree is untouched: remat preserves what param means,
    # so a bound node goes in and a bound node comes out
    assert jax.tree.structure(plain.param) == jax.tree.structure(remade.param)


def test_the_recomputation_is_actually_there():
    """A no-op wrapper would satisfy every value assertion above, so read
    the backward pass itself: the remat primitive is in the jaxpr of the
    rematerialized tower and absent from the plain one."""
    plain = str(jax.make_jaxpr(jax.grad(_loss))(_tower(_layer())))
    remade = str(jax.make_jaxpr(jax.grad(_loss))(_tower(remat(_layer()))))

    assert 'remat' not in plain
    assert 'remat' in remade


def test_remat_composes_where_depth_is_spelled():
    """The case it exists for: one block written once, stacked, batched,
    trained. remat sits at the layer, inside the stack that makes depth."""
    layer = remat(_layer())
    model = _tower(layer, n=6).initialize()
    trainer = train_step(model, lambda p, t: jnp.mean((p - t) ** 2), optax.adam(1e-2))
    steps = 20
    final, (_, aux) = trainer.scan(input=jnp.broadcast_to(X, (steps, *X.shape)),
                                   target=jnp.zeros((steps, 4, 8)))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]


def test_remat_is_a_wrapper_like_any_other():
    """Its recorded construction retains the wrapped node, so a structural
    rewrite reaches through it rather than stopping at the transform."""
    pipe = (nn.Linear(4) >> nn.EMA(0.1)).node
    model = batch(remat(pipe)).with_input(jnp.ones((3, 4))).parameterize(
        rng=jax.random.PRNGKey(0))

    state = model.init(input=jnp.ones((3, 4)))      # the EMA primes from a value
    frozen = tree_freeze(model, tree_filter(state, 'ema'))
    assert not frozen.cyclic                    # the only stateful member froze


def test_where_it_sits_is_the_granularity():
    """Being a node transform means PLACEMENT chooses what gets recomputed.
    Around a scan there are two useful answers and they are not the same
    trade: inside, the backward pass recomputes one step at a time and
    keeps the per-step boundaries; outside, it keeps only the sequence and
    recomputes the entire rollout. Both are exact — the gradients agree
    with the unrematerialized tower — so the choice is memory against
    recompute, nothing else."""
    seq = jnp.ones((16, 4))

    def grads(node):
        model = node.with_input(seq).parameterize(rng=jax.random.PRNGKey(0))
        loss = lambda m: jnp.sum(m.apply(seq) ** 2)
        return (jax.tree.leaves(jax.grad(loss)(model)),
                'remat' in str(jax.make_jaxpr(jax.grad(loss))(model)))

    plain, plain_has = grads(scanned(nn.RNN(4)))
    per_step, step_has = grads(scanned(remat(nn.RNN(4))))     # a step at a time
    rollout, roll_has = grads(remat(scanned(nn.RNN(4))))      # the whole sequence

    assert not plain_has and step_has and roll_has
    for a, b, c in zip(plain, per_step, rollout):
        assert jnp.allclose(a, b, atol=1e-4)
        assert jnp.allclose(a, c, atol=1e-4)


def test_a_substituted_member_reaches_the_apply():
    """The impl takes the wrapped def from the seam rather than closing over
    it, so swapping the member changes what runs. An impl closing over its
    def would keep running the original here, silently, and only look right
    because rebuild happens to re-run the transform."""
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
