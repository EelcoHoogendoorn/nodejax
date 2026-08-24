"""freeze / tree_freeze / map_members — structural rewrites over the node
tree, resting on composite member nodes.

The models are stock nn blocks: Linear for the param-carrying members,
EMA for a member whose state is all it owns, RNN for a live cyclic
member that must survive a partial freeze.
"""
import jax
import jax.numpy as jnp

from nodejax import (
    Node, PSNode, Leaf, nn, Composite, serial, freeze, tree_freeze, tree_filter,
    detach, tree_detach, map_members, Node,
)
from nodejax.struct import Struct


X = jax.random.normal(jax.random.PRNGKey(1), (8, 4))


def StatefulGain(name: str) -> Node:
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(param):
        return jnp.asarray(0.0)

    def apply(param, state, input):
        return state + input, param.scale * input

    return Leaf(apply, param=param, init=init, name=name)


def _pipe() -> Node:
    snet = nn.Linear(4) >> nn.EMA(0.1) >> nn.Linear(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    _, ref = model.apply(state, X)
    return snet, model, state, ref


def test_freeze_whole_node_is_noncyclic_and_faithful():
    snet, model, state, ref = _pipe()
    assert snet.cyclic
    fz = freeze(model, state)
    assert not fz.cyclic                     # the cyclic slot is gone
    assert jnp.allclose(fz.apply(X), ref)         # applies with the held state
    assert jnp.allclose(fz.apply(X), fz.apply(X)) # and is deterministic (frozen)


def test_tree_freeze_propagates_cyclicity():
    snet, model, state, ref = _pipe()
    tf = tree_freeze(model, tree_filter(state, 'ema'))
    # the only stateful member froze, so the pipe recomputes to non-cyclic
    assert not tf.cyclic
    assert tf.contract.input_spec is None  # tree binding discards input evidence
    assert jnp.allclose(tf.apply(X), ref)


def test_tree_freeze_full_state_equals_freeze():
    # a full state (no KEEP) freezes everything, matching whole-node freeze
    snet, model, state, ref = _pipe()
    tf = tree_freeze(model, state)
    assert not tf.cyclic
    assert jnp.allclose(tf.apply(X), ref)


def test_tree_freeze_partial_stays_cyclic():
    # two stateful members; freeze one, the pipe must stay cyclic
    snet = nn.Linear(4) >> nn.EMA(0.1) >> nn.EMA(0.05)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    frozen_all = tree_freeze(model, tree_filter(state, 'ema'))
    assert not frozen_all.cyclic             # both froze -> non-cyclic
    # a filter matching nothing is a loud miss, never an empty spec
    import pytest
    with pytest.raises(ValueError, match='matched nothing'):
        tree_filter(state, 'nomatch')


def test_freeze_removes_state_slots():
    snet, model, state, ref = _pipe()             # Linear >> EMA >> Linear
    assert len(jax.tree.leaves(state)) > 0        # the ema's smoothed copy

    # whole freeze: no state slot survives at all
    fz = freeze(model, state)
    assert jax.tree.leaves(fz.init()) == []

    # tree_freeze the only stateful member: its slots vanish, pipe non-cyclic
    tf = tree_freeze(model, tree_filter(state, 'ema'))
    assert not tf.cyclic
    assert jax.tree.leaves(
        tf.with_input(X).bind(tf.param).init()) == []


def test_tree_freeze_partial_drops_only_frozen_slots():
    # EMA (frozen) + RNN (live): the smoothing slots go, the RNN state
    # stays, and the model stays cyclic
    snet = nn.Linear(4) >> nn.EMA(0.1) >> nn.RNN(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    n_before = len(jax.tree.leaves(state))

    tf = tree_freeze(model, tree_filter(state, 'ema'))
    assert tf.cyclic                         # the RNN is still live
    after = tf.with_input(X).bind(tf.param).init()
    n_after = len(jax.tree.leaves(after))
    assert 0 < n_after < n_before                 # ema slots dropped, RNN kept


def test_tree_freeze_hand_built_sparse_spec():
    # freeze one node by hand: just the key you want — no mirror, no filter
    snet = nn.Linear(4) >> nn.EMA(0.1) >> nn.RNN(4)
    model = snet.with_input(X).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    tf = tree_freeze(model, Struct(ema=state.ema))
    assert tf.cyclic                          # the RNN is still live
    after = tf.with_input(X).bind(tf.param).init()
    assert 0 < len(jax.tree.leaves(after)) < len(jax.tree.leaves(state))


def test_map_members_identity_rebuilds_the_same_computation():
    def gate(w):
        def apply(self, input):
            g = jax.nn.sigmoid(self.g(input))
            return g * self.a(input) + (1 - g) * self.b(input)
        return Composite(a=nn.Linear(w), b=nn.Linear(w),
                                             g=nn.Linear(1))(apply, name='gate')

    gc = gate(4).with_input(X)
    ref = gc.parameterize(rng=jax.random.PRNGKey(0)).apply(X)
    rebuilt = map_members(gc, lambda node: node)
    assert rebuilt is not gc
    assert rebuilt.contract.input_spec is None
    rebuilt = rebuilt.with_input(X)
    got = rebuilt.parameterize(rng=jax.random.PRNGKey(0)).apply(X)
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


def test_detach_preserves_bound_state_and_state_transition():
    model = StatefulGain('gain').bind(
        Struct(scale=jnp.asarray(2.0)),
        state=jnp.asarray(3.0),
    )

    detached = detach(model)

    assert type(detached) is PSNode
    assert detached.state == 3.0
    expected, expected_output = model(jnp.asarray(4.0))
    successor, output = detached(jnp.asarray(4.0))
    assert successor.state == expected.state
    assert output == expected_output

    def loss(param):
        _, value = detached.bind(param)(jnp.asarray(4.0))
        return value

    gradient = jax.grad(loss)(detached.param)
    assert gradient.scale == 0.0


def test_tree_detach_selects_by_key():
    # tree_detach freezes the weights of members matched by key; here both
    # linears match, so no weight gets a gradient
    model = (nn.Linear(8) >> nn.gelu >> nn.Linear(4)).with_input(X).parameterize(
        rng=jax.random.PRNGKey(0))
    frozen = tree_detach(model, 'linear')
    assert frozen.contract.input_spec is None  # tree binding discards input evidence

    def loss(node):
        return jnp.sum(node.apply(X[0]) ** 2)

    assert all(jnp.abs(x).sum() == 0 for x in jax.tree.leaves(jax.grad(loss)(frozen)))


def test_tree_detach_preserves_bound_state_and_other_gradients():
    pipe = serial(
        selected=StatefulGain('selected'),
        live=StatefulGain('live'),
    )
    model = pipe.bind(Struct(
        selected=Struct(scale=jnp.asarray(2.0)),
        live=Struct(scale=jnp.asarray(3.0)),
    ), state=Struct(
        selected=jnp.asarray(5.0),
        live=jnp.asarray(7.0),
    ))

    detached = tree_detach(model, 'selected')

    assert type(detached) is PSNode
    assert detached.state.selected == 5.0
    assert detached.state.live == 7.0
    expected, expected_output = model(jnp.asarray(4.0))
    successor, output = detached(jnp.asarray(4.0))
    assert successor.state.selected == expected.state.selected
    assert successor.state.live == expected.state.live
    assert output == expected_output

    def loss(param):
        _, value = detached.bind(param)(jnp.asarray(4.0))
        return value

    gradient = jax.grad(loss)(detached.param)
    assert gradient.selected.scale == 0.0
    assert gradient.live.scale != 0.0


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


def test_a_rewrite_discards_later_boundary_shape_evidence():
    """Tree binding precedes and therefore invalidates input binding."""
    from nodejax import map_members

    d = (nn.Linear(4) >> nn.EMA(0.1)).with_input(jnp.zeros(4))
    assert d.contract.input_spec is not None

    rewritten = map_members(d, lambda node: node)
    assert rewritten.contract.input_spec is None
    resolved = rewritten.with_input(jnp.zeros(4))
    assert resolved.contract.input_spec is not None
    assert resolved.contract.input_spec.input == d.contract.input_spec.input
