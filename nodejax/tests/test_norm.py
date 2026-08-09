"""Batchnorm as ordinary cyclic state, per-sample under a named axis.

Norm needs no mode flags: running stats are just cyclic state; eval =
freezing the state. The node is written per-sample, its moments
collectives over the NAMED batch axis, so it declares the axis need
and batch() binds it — an unbatched norm refuses to bind at all.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import Node, NodeDef, batch, train_step, nn
from nodejax.struct import Struct
from nodejax.util import mse, tile


def test_nn_batch_norm():
    """Verify nn.BatchNorm in nn module has single_batch_state tag and updates running stats."""
    bn = nn.BatchNorm(momentum=0.1)
    assert isinstance(bn, NodeDef) and bn.parametric and bn.cyclic
    assert 'single_batch_state' in bn.tags

    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    model = batch(nn.Linear(4) >> bn).with_input(x).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    assert state.bn.mean.shape == (4,)
    state, out = model.apply(state, x)
    assert state.bn.mean.shape == (4,)


def test_norm_running_stats_converge():
    """A batch of three scalar samples: every element's collective sees
    the same batch moments, so the per-element stats agree and converge
    to the population's."""
    x = jnp.array([[3.0], [5.0], [7.0]])
    cn = batch(nn.BatchNorm(momentum=0.1)).with_input(jnp.zeros_like(x)).parameterize()

    state, _ = cn.scan(None, tile(x, 300))
    assert jnp.allclose(state.mean, jnp.mean(x), atol=0.01)
    assert jnp.allclose(state.var, jnp.var(x), atol=0.01)

    # converged stats -> normalized output
    _, out = cn.apply(state, x)
    assert jnp.allclose(jnp.mean(out), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out), 1.0, atol=0.05)


def test_linear_norm_pipe():
    """Linear >> Norm, per-sample; batch() binds the axis for the whole
    pipe, and the running stats converge to the batch statistics."""
    F = 4
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    assert isinstance(pipe, NodeDef) and pipe.parametric and pipe.cyclic

    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0]])
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=5.0 * jnp.eye(F), b=10.0 * jnp.ones(F)),
        bn=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    state = bound.init()

    # scan over 300 identical batches to converge the running stats
    state, outputs = bound.scan(state, tile(x, 300))
    assert outputs.shape == (300, 3, F)
    expected_mean = jnp.mean(5.0 * x + 10.0, axis=0)
    assert jnp.allclose(state.bn.mean, expected_mean, atol=0.1)

    # after convergence the output is per-feature normalized
    _, out = bound.apply(state, x)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out, axis=0), 1.0, atol=0.05)


def test_linear_norm_eval_frozen_state():
    """Eval mode = reusing a frozen state; no flags anywhere."""
    F = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=2.0 * jnp.eye(F), b=jnp.ones(F)),
        bn=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    trained, _ = bound.scan(bound.init(), tile(x, 200))

    _, out1 = bound.apply(trained, x)
    _, out2 = bound.apply(trained, x)
    _, out3 = bound.apply(trained, x + 1.0)
    assert jnp.allclose(out1, out2)          # frozen state -> deterministic
    assert not jnp.allclose(out1, out3)      # but still a function of input


def test_linear_norm_dimension_change():
    """Norm param shape couples to Linear out_features (4 -> 3)."""
    x = jnp.arange(20.0).reshape(5, 4)
    pipe = nn.Linear(3) >> nn.BatchNorm(momentum=0.1)
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=jnp.ones((4, 3)), b=jnp.zeros(3)),
        bn=Struct(gamma=jnp.ones(3), beta=jnp.zeros(3)),
    ))
    state, _ = bound.scan(bound.init(), tile(x, 300))
    _, out = bound.apply(state, x)
    assert out.shape == (5, 3)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)


def test_train_linear_norm():
    """Training a batchnormed model is just train_step(batch(pipe)): the
    running stats update during training because they are state, not
    because of a training-mode flag."""
    F = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    batched = batch(pipe).with_input(jnp.zeros_like(x))
    trainer = train_step(batched, mse, optax.adam(0.05))

    model = batched.bind(Struct(
        linear=Struct(w=jnp.eye(F), b=jnp.zeros(F)),
        bn=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    state = trainer.init(model=model.param)

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
    pre_norm = x @ final.model.linear.w + final.model.linear.b
    batch_mean = jnp.mean(pre_norm, axis=0)
    running = final.inner.bn.mean     # single_batch_state: unbatched 1D stats array
    assert jnp.linalg.norm(running) > 5 * jnp.linalg.norm(0.1 * batch_mean)


def test_nn_whiten():
    """Verify nn.Whiten computes running covariance as single_batch_state."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    node = batch(nn.Whiten(momentum=0.1)).with_input(jnp.zeros_like(x))
    state = node.init(input=x)
    assert state.mean.shape == (2,)
    assert state.cov.shape == (2, 2)

    new_state, out = node.apply(state, x)
    assert out.shape == (2, 2)
    assert new_state.mean.shape == (2,)
    assert new_state.cov.shape == (2, 2)
