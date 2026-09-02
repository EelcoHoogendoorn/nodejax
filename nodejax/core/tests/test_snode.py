"""PSNode: the state-bound rung of the binding ladder.

(definition, param, state) as one object. It owns its state: apply takes no
state and returns none, and the advance flows into the successor,
session, out = session(x), one arity whatever the node's kind. A pytree with
param and state as separate children, so lax.scan carries a whole
PSNode. trained() lands here: a finished run hands back the model as a
thing you can call.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import PNode, PSNode, nn, Leaf, train_step, trained, tree_freeze
from nodejax.control import Integrator
from nodejax.core.psnode import _session_scan
from nodejax.struct import Struct


def test_the_successor_carries_the_advance():
    session = Integrator().parameterize().bind(state=jnp.asarray(0.0))
    successor, out = session(1.0)
    sn3, out2 = successor(1.0)
    assert type(successor) is PSNode
    assert jnp.allclose(sn3.state, 2.0)
    assert jnp.allclose(session.state, 0.0)          # the predecessor stands


def test_one_arity_and_the_acyclic_successor():
    session = nn.gelu.bind(state=())
    successor, out = session(jnp.ones(3))
    assert successor.state == ()
    assert jnp.allclose(out, jax.nn.gelu(jnp.ones(3)))


def test_scan_carries_a_whole_snode():
    start = Integrator().parameterize().bind(state=jnp.asarray(0.0))
    final, ys = jax.lax.scan(lambda session, x: session(x), start, jnp.ones(5))
    assert type(final) is PSNode
    assert jnp.allclose(final.state, 5.0)
    assert ys.shape == (5,)


def test_session_scan_reuses_one_compilation():
    _session_scan.clear_cache()
    session = Integrator().parameterize().bind(
        state=jnp.zeros((), dtype=jnp.float32))

    try:
        for _ in range(3):
            session, _ = session.scan(jnp.ones(2))
        assert _session_scan._cache_size() == 1
        assert jnp.allclose(session.state, 6.0)
    finally:
        _session_scan.clear_cache()


def test_methods_bind_the_state_seat():
    """A node method naming `state` reads the live state on a PSNode,
    where a bare PNode leaves that channel to the caller."""
    def Level():
        def apply(state, input):
            return state + input, state
        def level(state):
            return state
        return Leaf(apply, init=lambda: jnp.zeros(()), name='level',
                        methods=dict(level=level))

    session = Level().bind(state=jnp.asarray(3.0))
    assert jnp.allclose(session.level(), 3.0)


def test_trained_hands_back_a_callable_model():
    """The finalizing projection lands as a PSNode of the MODEL: weights
    at .param, the model's own state carried, the trainer scaffolding
    struck. done(x) infers; done.pnode is the param-only view."""
    key = jax.random.PRNGKey(0)
    net = nn.Linear(1).with_input(jnp.zeros(3)).parameterize(rng=key).initialize()
    trainer = train_step(net, lambda out, target: jnp.mean((out - target) ** 2),
                         optax.sgd(0.1))
    xs = jax.random.normal(key, (64, 3))
    w_true = jnp.asarray([[1.0], [-2.0], [0.5]])
    done, aux = trained(trainer).apply(input=xs, target=xs @ w_true)

    assert type(done) is PSNode
    assert jnp.allclose(done.param.w, w_true, atol=0.1)
    _, pred = done(jnp.ones(3))
    assert pred.shape == (1,)
    assert type(done.pnode) is PNode


def test_a_whole_tree_freeze_makes_a_plain_function():
    """tree_freeze consumes the binding it holds: no spec argument, the
    state comes from the object, and a whole-tree freeze leaves nothing
    moving, so the successor equals its predecessor and discarding it is
    safe."""
    pipe = (Integrator() >> Integrator()).parameterize()
    session = pipe.bind(state=pipe.init())
    successor, _ = session(1.0)
    frozen = tree_freeze(successor)
    assert type(frozen) is PSNode and frozen.state == ()
    f2, a = frozen(1.0)
    f3, b = frozen(1.0)
    assert jnp.allclose(a, b)                    # pinned state: same answer
    assert f2.state == () == f3.state
