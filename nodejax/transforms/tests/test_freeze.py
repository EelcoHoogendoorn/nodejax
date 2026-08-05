"""freeze / tree_freeze / map_members — structural rewrites over the def
tree, resting on the reconstructable-def recipe."""
import jax
import jax.numpy as jnp

from nodejax import (node_def, composite, freeze, tree_freeze, tree_filter,
                           detach, tree_detach, map_members, NodeDef)
from nodejax.authoring import KeyStream
from nodejax.struct import Struct


def dense(n_in, n_out):
    def param(rng: KeyStream):
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))

    def apply(param, input):
        return input @ param.w + param.b

    return node_def(apply, param=param, name='dense')


gelu = node_def(lambda input: jax.nn.gelu(input), name='gelu')


def running_norm(momentum):
    def init(ndef):
        f = jax.tree.map(lambda x: x[0], ndef.input)
        return Struct(mean=jnp.zeros_like(f), var=jnp.ones_like(f))

    def apply(state, input):
        out = (input - state.mean) / jnp.sqrt(state.var + 1e-5)
        new = Struct(mean=(1 - momentum) * state.mean + momentum * jnp.mean(input, 0),
                     var=(1 - momentum) * state.var + momentum * jnp.var(input, 0))
        return new, out

    return node_def(apply, init=init, name='running_norm')


def rnn(width):
    def param(rng: KeyStream):
        return Struct(wh=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width))

    def init(param, ndef):
        return jnp.zeros_like(ndef.input)

    def apply(param, state, input):
        h = jnp.tanh(input + state @ param.wh)
        return h, h

    return node_def(apply, init=init, param=param, name='rnn')


X = jax.random.normal(jax.random.PRNGKey(1), (8, 4))


def _pipe():
    snet = dense(4, 4) >> running_norm(0.1) >> dense(4, 4)
    model = snet.parameterize(rng=jax.random.PRNGKey(0))
    state = model.with_input(X).init()
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
    tf = tree_freeze(model, tree_filter(state, 'running'))
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
    snet = dense(4, 4) >> running_norm(0.1) >> running_norm(0.05)
    model = snet.parameterize(rng=jax.random.PRNGKey(0))
    state = model.with_input(X).init()
    frozen_all = tree_freeze(model, tree_filter(state, 'running'))
    assert not frozen_all.ndef.cyclic             # both froze -> non-cyclic
    # a filter matching nothing is a loud miss, never an empty spec
    import pytest
    with pytest.raises(ValueError, match='matched nothing'):
        tree_filter(state, 'nomatch')


def test_freeze_removes_state_slots():
    snet, model, state, ref = _pipe()             # dense >> running_norm >> dense
    assert len(jax.tree.leaves(state)) > 0        # the running-norm's mean/var

    # whole freeze: no state slot survives at all
    fz = freeze(model, state)
    assert jax.tree.leaves(fz.init()) == []

    # tree_freeze the only stateful member: its slots vanish, pipe non-cyclic
    tf = tree_freeze(model, tree_filter(state, 'running'))
    assert not tf.ndef.cyclic
    assert jax.tree.leaves(tf.with_input(X).init()) == []


def test_tree_freeze_partial_drops_only_frozen_slots():
    # running_norm (frozen) + rnn (live): the running-norm slots go, the
    # rnn state stays, and the model stays cyclic
    snet = dense(4, 4) >> running_norm(0.1) >> rnn(4)
    model = snet.parameterize(rng=jax.random.PRNGKey(0))
    state = model.with_input(X).init()
    n_before = len(jax.tree.leaves(state))

    tf = tree_freeze(model, tree_filter(state, 'running'))
    assert tf.ndef.cyclic                         # the rnn is still live
    after = tf.with_input(X).init()
    n_after = len(jax.tree.leaves(after))
    assert 0 < n_after < n_before                 # norm slots dropped, rnn kept
    # concretely: no mean/var left, but the rnn hidden state is there
    names = {p[-1].name if hasattr(p[-1], 'name') else p[-1]
             for p, _ in jax.tree_util.tree_flatten_with_path(after)[0]}
    assert 'mean' not in names and 'var' not in names


def test_tree_freeze_hand_built_sparse_spec():
    # freeze one node by hand: just the key you want — no mirror, no filter
    snet = dense(4, 4) >> running_norm(0.1) >> rnn(4)
    model = snet.parameterize(rng=jax.random.PRNGKey(0))
    state = model.with_input(X).init()
    tf = tree_freeze(model, Struct(running_norm=state.running_norm))
    assert tf.ndef.cyclic                          # the rnn is still live
    after = tf.with_input(X).init()
    assert 0 < len(jax.tree.leaves(after)) < len(jax.tree.leaves(state))


def test_map_members_identity_preserves_composite():
    def gate(w):
        def apply(self, input):
            g = jax.nn.sigmoid(self.g(input))
            return g * self.a(input) + (1 - g) * self.b(input)
        return composite(apply, members=dict(a=dense(w, w), b=dense(w, w), g=dense(w, 1)),
                         name='gate')

    gc = gate(4)
    ref = gc.parameterize(rng=jax.random.PRNGKey(0)).apply(X)
    rebuilt = map_members(gc, lambda nd: nd)      # identity rewrite = full rebuild
    got = rebuilt.parameterize(rng=jax.random.PRNGKey(0)).apply(X)
    assert jnp.allclose(ref, got)


def test_detach_stops_gradient():
    # detach is the param-side twin of freeze: it pins weights, not state
    model = (dense(4, 8) >> gelu >> dense(8, 4)).parameterize(rng=jax.random.PRNGKey(0))

    def loss(node):
        return jnp.sum(node.apply(X[0]) ** 2)

    g_full = jax.tree.leaves(jax.grad(loss)(model))
    g_detached = jax.tree.leaves(jax.grad(loss)(detach(model)))
    assert any(jnp.abs(x).sum() > 0 for x in g_full)       # baseline has real grads
    assert all(jnp.abs(x).sum() == 0 for x in g_detached)  # detached weights get none


def test_tree_detach_selects_by_key():
    # tree_detach freezes the weights of members matched by key; here both
    # denses match, so no weight gets a gradient
    model = (dense(4, 8) >> gelu >> dense(8, 4)).parameterize(rng=jax.random.PRNGKey(0))
    frozen = tree_detach(model, 'dense')

    def loss(node):
        return jnp.sum(node.apply(X[0]) ** 2)

    assert all(jnp.abs(x).sum() == 0 for x in jax.tree.leaves(jax.grad(loss)(frozen)))


def test_selective_walks_fail_loud_on_a_miss():
    """A selector that lands on nothing raises at the call: leaf targets
    point to the whole-node twins, and a name matching no member is an
    error, never a silent identity."""
    import pytest
    model = (dense(4, 8) >> gelu >> dense(8, 4)).parameterize(rng=jax.random.PRNGKey(0))
    leaf = dense(4, 4).parameterize(rng=jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match='matched no member'):
        tree_detach(model, 'enc')
    with pytest.raises(TypeError, match='detach\\(node\\)'):
        tree_detach(leaf, 'dense')
    with pytest.raises(TypeError, match='name no member'):
        tree_freeze(model, Struct(enc=Struct()))
    with pytest.raises(TypeError, match='freeze\\(node, state\\)'):
        tree_freeze(leaf, Struct())
    with pytest.raises(ValueError, match='matched nothing'):
        tree_filter(Struct(a=Struct(x=1.0)), 'enc')
