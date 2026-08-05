"""Batchnorm as ordinary cyclic state.

Norm needs no mode flags: running stats are just cyclic state; eval =
freezing the state.
"""

import jax.numpy as jnp
import optax

from nodejax import Node, NodeDef, train_step
from nodejax.struct import Struct
from nodejax.examples import Linear, Norm, mse, tile


def test_norm_types():
    bn = Norm(momentum=0.1, shape=(1,))
    assert isinstance(bn, NodeDef) and bn.parametric and bn.cyclic
    cn = bn.parameterize(gamma=1.0, beta=0.0)
    assert isinstance(cn, Node) and cn.cyclic


def test_norm_running_stats_converge():
    cn = Norm(momentum=0.1, shape=(1,)).parameterize(gamma=1.0, beta=0.0)
    x = jnp.array([3.0, 5.0, 7.0])

    state, _ = cn.scan(None, tile(x, 300))
    assert jnp.allclose(state.running_mean, jnp.mean(x), atol=0.01)
    assert jnp.allclose(state.running_var, jnp.var(x), atol=0.01)

    # converged stats -> normalized output
    _, out = cn.apply(state, x)
    assert jnp.allclose(jnp.mean(out), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out), 1.0, atol=0.05)


def test_linear_norm_pipe():
    """Linear >> Norm: specialize, compose, converge the running stats."""
    F = 4
    pipe = Linear(F, F) >> Norm(momentum=0.1, shape=(1, F))
    assert isinstance(pipe, NodeDef) and pipe.parametric and pipe.cyclic

    bound = pipe.parameterize(
        linear4x4=Struct(weight=5.0 * jnp.eye(F), bias=10.0 * jnp.ones(F)),
        norm=Struct(gamma=jnp.ones((1, F)), beta=jnp.zeros((1, F))),
    )
    state = bound.init()
    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0]])

    # scan over 300 identical batches to converge the running stats
    state, outputs = bound.scan(state, tile(x, 300))
    assert outputs.shape == (300, 3, F)
    expected_mean = jnp.mean(5.0 * x + 10.0, axis=0, keepdims=True)
    assert jnp.allclose(state.norm.running_mean, expected_mean, atol=0.1)

    # after convergence the output is per-feature normalized
    _, out = bound.apply(state, x)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out, axis=0), 1.0, atol=0.05)


def test_linear_norm_eval_frozen_state():
    """Eval mode = reusing a frozen state; no flags anywhere."""
    F = 4
    pipe = Linear(F, F) >> Norm(momentum=0.1, shape=(1, F))
    bound = pipe.parameterize(
        linear4x4=Struct(weight=2.0 * jnp.eye(F), bias=jnp.ones(F)),
        norm=Struct(gamma=jnp.ones((1, F)), beta=jnp.zeros((1, F))),
    )
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    trained, _ = bound.scan(bound.init(), tile(x, 200))

    _, out1 = bound.apply(trained, x)
    _, out2 = bound.apply(trained, x)
    _, out3 = bound.apply(trained, x + 1.0)
    assert jnp.allclose(out1, out2)          # frozen state -> deterministic
    assert not jnp.allclose(out1, out3)      # but still a function of input


def test_linear_norm_dimension_change():
    """Norm shape couples to Linear out_features (4 -> 3)."""
    pipe = Linear(4, 3) >> Norm(momentum=0.1, shape=(1, 3))
    bound = pipe.parameterize(
        linear4x3=Struct(weight=jnp.ones((4, 3)), bias=jnp.zeros(3)),
        norm=Struct(gamma=jnp.ones((1, 3)), beta=jnp.zeros((1, 3))),
    )
    x = jnp.arange(20.0).reshape(5, 4)
    state, _ = bound.scan(bound.init(), tile(x, 300))
    _, out = bound.apply(state, x)
    assert out.shape == (5, 3)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)


def test_train_linear_norm():
    """Training a batchnormed model is just train_step(pipe): the running
    stats update during training because they are state, not because of a
    training-mode flag."""
    F = 4
    pipe = Linear(F, F) >> Norm(momentum=0.1, shape=(1, F))
    trainer = train_step(pipe, mse, optax.adam(0.05))

    model = pipe.parameterize(
        linear4x4=Struct(weight=jnp.eye(F), bias=jnp.zeros(F)),
        norm=Struct(gamma=jnp.ones((1, F)), beta=jnp.zeros((1, F))),
    )
    state = trainer.init(model=model.param)

    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    steps = 200
    inputs = Struct(input=tile(x, steps), target=jnp.zeros((steps, 2, F)))
    final, losses = trainer.scan(state, inputs)

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]
    # the running stats THREADED through all 200 steps: they carry
    # full-scale history of the pre-norm activations (an EMA trailing
    # the still-moving weights). A state re-initialized every step
    # would sit at momentum * batch_mean — an order of magnitude
    # smaller and the decisive signature of broken threading.
    pre_norm = x @ final.model.linear4x4.weight + final.model.linear4x4.bias
    batch_mean = jnp.mean(pre_norm, axis=0, keepdims=True)
    running = final.inner.norm.running_mean
    assert jnp.linalg.norm(running) > 5 * jnp.linalg.norm(0.1 * batch_mean)
    assert jnp.all(jnp.sign(running) == jnp.sign(batch_mean))
