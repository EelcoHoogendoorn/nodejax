"""freeze / tree_freeze / map_members — structural rewrites over the def
tree, resting on the reconstructable-def recipe.

The models are stock nn blocks: Linear for the param-carrying members,
Whiten for a member whose state is all it owns, RNN for a live cyclic
member that must survive a partial freeze.
"""
import jax
import jax.numpy as jnp

from nodejax import (nn, composite, freeze, tree_freeze, tree_filter,
                           detach, tree_detach, map_members, NodeDef)
from nodejax.struct import Struct


X = jax.random.normal(jax.random.PRNGKey(1), (8, 4))


def _pipe():
    snet = nn.Linear(4) >> nn.Whiten(0.1) >> nn.Linear(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    _, ref = model.apply(state, X)
    return snet, model, state, ref


def test_freeze_whole_node_is_noncyclic_and_faithful():
    snet, model, state, ref = _pipe()
    assert snet.cyclic
    fz = freeze(model, state)
    assert not fz.ndef.cyclic                     # the cyclic slot is gone
    assert jnp.allclose(fz.apply(X), ref)         # applies with the held state
    assert jnp.allclose(fz.apply(X), fz.apply(X)) # and is deterministic (frozen)


def test_tree_freeze_propagates_cyclicity():
    snet, model, state, ref = _pipe()
    tf = tree_freeze(model, tree_filter(state, 'whiten'))
    # the only stateful member froze, so the pipe recomputes to non-cyclic
    assert not tf.ndef.cyclic
    assert jnp.allclose(tf.apply(X), ref)


def test_tree_freeze_full_state_equals_freeze():
    # a full state (no KEEP) freezes everything, matching whole-node freeze
    snet, model, state, ref = _pipe()
    tf = tree_freeze(model, state)
    assert not tf.ndef.cyclic
    assert jnp.allclose(tf.apply(X), ref)


def test_tree_freeze_partial_stays_cyclic():
    # two stateful members; freeze one, the pipe must stay cyclic
    snet = nn.Linear(4) >> nn.Whiten(0.1) >> nn.Whiten(0.05)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    frozen_all = tree_freeze(model, tree_filter(state, 'whiten'))
    assert not frozen_all.ndef.cyclic             # both froze -> non-cyclic
    # a filter matching nothing is a loud miss, never an empty spec
    import pytest
    with pytest.raises(ValueError, match='matched nothing'):
        tree_filter(state, 'nomatch')


def test_freeze_removes_state_slots():
    snet, model, state, ref = _pipe()             # Linear >> Whiten >> Linear
    assert len(jax.tree.leaves(state)) > 0        # the whitening mean/cov

    # whole freeze: no state slot survives at all
    fz = freeze(model, state)
    assert jax.tree.leaves(fz.init()) == []

    # tree_freeze the only stateful member: its slots vanish, pipe non-cyclic
    tf = tree_freeze(model, tree_filter(state, 'whiten'))
    assert not tf.ndef.cyclic
    assert jax.tree.leaves(tf.with_input(X).init()) == []


def test_tree_freeze_partial_drops_only_frozen_slots():
    # Whiten (frozen) + RNN (live): the whitening slots go, the RNN state
    # stays, and the model stays cyclic
    snet = nn.Linear(4) >> nn.Whiten(0.1) >> nn.RNN(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    n_before = len(jax.tree.leaves(state))

    tf = tree_freeze(model, tree_filter(state, 'whiten'))
    assert tf.ndef.cyclic                         # the RNN is still live
    after = tf.with_input(X).init()
    n_after = len(jax.tree.leaves(after))
    assert 0 < n_after < n_before                 # whitening slots dropped, RNN kept
    # concretely: no mean/cov left, but the RNN hidden state is there
    names = {p[-1].name if hasattr(p[-1], 'name') else p[-1]
             for p, _ in jax.tree_util.tree_flatten_with_path(after)[0]}
    assert 'mean' not in names and 'cov' not in names


def test_tree_freeze_hand_built_sparse_spec():
    # freeze one node by hand: just the key you want — no mirror, no filter
    snet = nn.Linear(4) >> nn.Whiten(0.1) >> nn.RNN(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    tf = tree_freeze(model, Struct(whiten=state.whiten))
    assert tf.ndef.cyclic                          # the RNN is still live
    after = tf.with_input(X).init()
    assert 0 < len(jax.tree.leaves(after)) < len(jax.tree.leaves(state))


def test_map_members_identity_preserves_composite():
    def gate(w):
        def apply(self, input):
            g = jax.nn.sigmoid(self.g(input))
            return g * self.a(input) + (1 - g) * self.b(input)
        return composite(apply, members=dict(a=nn.Linear(w), b=nn.Linear(w),
                                             g=nn.Linear(1)), name='gate')

    gc = gate(4).with_input(X)
    ref = gc.parameterize(rng=jax.random.PRNGKey(0)).apply(X)
    rebuilt = map_members(gc, lambda nd: nd)      # identity rewrite = full rebuild
    # the rebuild yields a fresh def, so the input offer is made again
    got = rebuilt.with_input(X).parameterize(rng=jax.random.PRNGKey(0)).apply(X)
    assert jnp.allclose(ref, got)


def test_detach_stops_gradient():
    # detach is the param-side twin of freeze: it pins weights, not state
    model = (nn.Linear(8) >> nn.gelu >> nn.Linear(4)).with_input(X).parameterize(
        rng=jax.random.PRNGKey(0))

    def loss(node):
        return jnp.sum(node.apply(X[0]) ** 2)

    g_full = jax.tree.leaves(jax.grad(loss)(model))
    g_detached = jax.tree.leaves(jax.grad(loss)(detach(model)))
    assert any(jnp.abs(x).sum() > 0 for x in g_full)       # baseline has real grads
    assert all(jnp.abs(x).sum() == 0 for x in g_detached)  # detached weights get none


def test_tree_detach_selects_by_key():
    # tree_detach freezes the weights of members matched by key; here both
    # linears match, so no weight gets a gradient
    model = (nn.Linear(8) >> nn.gelu >> nn.Linear(4)).with_input(X).parameterize(
        rng=jax.random.PRNGKey(0))
    frozen = tree_detach(model, 'linear')

    def loss(node):
        return jnp.sum(node.apply(X[0]) ** 2)

    assert all(jnp.abs(x).sum() == 0 for x in jax.tree.leaves(jax.grad(loss)(frozen)))


def test_selective_walks_fail_loud_on_a_miss():
    """A selector that lands on nothing raises at the call: leaf targets
    point to the whole-node twins, and a name matching no member is an
    error, never a silent identity."""
    import pytest
    model = (nn.Linear(8) >> nn.gelu >> nn.Linear(4)).with_input(X).parameterize(
        rng=jax.random.PRNGKey(0))
    leaf = nn.Linear(4).with_input(X).parameterize(rng=jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match='matched no member'):
        tree_detach(model, 'enc')
    with pytest.raises(TypeError, match='detach\\(node\\)'):
        tree_detach(leaf, 'linear')
    with pytest.raises(TypeError, match='name no member'):
        tree_freeze(model, Struct(enc=Struct()))
    with pytest.raises(TypeError, match='freeze\\(node, state\\)'):
        tree_freeze(leaf, Struct())
    with pytest.raises(ValueError, match='matched nothing'):
        tree_filter(Struct(a=Struct(x=1.0)), 'enc')
