"""Verifies `PSNode` state-bound execution, JAX pytree scanning, and `trained()` projection.

`PSNode` pairs a definition with bound `param` and `state` trees. Calling a
`PSNode` advances state into a new `PSNode` successor alongside the output,
registers `param` and `state` as separate JAX pytree children for `lax.scan`,
and serves as the callable model returned by `trained()`.
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
    """Verifies a node method declaring `state` reads the bound state on `PSNode`."""
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
    """Verifies `trained(trainer)` returns an acyclic `PNode` carrying the fitted weights."""
    key = jax.random.PRNGKey(0)
    net = nn.Linear(1).with_input(jnp.zeros(3)).parameterize(rng=key).initialize()
    trainer = train_step(net, lambda out, target: jnp.mean((out - target) ** 2),
                         optax.sgd(0.1))
    xs = jax.random.normal(key, (64, 3))
    w_true = jnp.asarray([[1.0], [-2.0], [0.5]])
    done, aux = trained(trainer).apply(input=xs, target=xs @ w_true)

    assert type(done) is PNode
    assert jnp.allclose(done.param.w, w_true, atol=0.1)
    pred = done(jnp.ones(3))
    assert pred.shape == (1,)
    assert type(done.pnode) is PNode


def test_trained_returns_psnode_for_cyclic_model():
    """Verifies `trained(trainer)` returns a `PSNode` carrying live state for a cyclic model."""
    model = Integrator().parameterize(decay=0.1).initialize()
    trainer = train_step(model, lambda out, target: (out - target) ** 2,
                         optax.sgd(0.05))
    done, _ = trained(trainer).apply(input=jnp.ones(4), target=jnp.ones(4))
    assert type(done) is PSNode and done.cyclic
    successor, out = done(1.0)
    assert type(successor) is PSNode and out.shape == ()


def test_a_whole_tree_freeze_makes_a_plain_function():
    """Verifies `tree_freeze(psnode)` embeds the bound state and returns an acyclic `PNode`."""
    pipe = (Integrator() >> Integrator()).parameterize()
    session = pipe.bind(state=pipe.init())
    successor, _ = session(1.0)
    frozen = tree_freeze(successor)
    assert type(frozen) is PNode and not frozen.cyclic
    a = frozen(1.0)
    b = frozen(1.0)
    assert jnp.allclose(a, b)                    # frozen state: same answer
