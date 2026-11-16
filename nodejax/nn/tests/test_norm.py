"""Verifies `BatchNorm` and `Whiten` as per-sample cyclic Nodes over named collective axes.

Running statistics live in cyclic `state`, so evaluation freezes `state` rather
than toggling a mode flag. Collective moments reduce over named mapped axes
bound by `batch` (or positional sample axes), refusing unbound collective axes.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import scan, scanned, PNode, Node, batch, train_step, nn
from nodejax.struct import Struct
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


from nodejax import tree_freeze

def test_batchnorm_unbatched_single_sample_eval():
    """Verifies single-sample evaluation on unbatched `BatchNorm` when its state is frozen."""
    bn = nn.BatchNorm(momentum=0.1)
    pipe = (nn.Linear(4) >> bn).with_input(jnp.zeros(4))
    bound_pipe = pipe.parameterize(rng=jax.random.PRNGKey(0))

    # Freezing the state-bound pipe holds the running statistics at their initial values.
    frozen = tree_freeze(bound_pipe.initialize())
    _, sample_out = frozen(jnp.ones(4))
    assert sample_out.shape == (4,)


def test_nn_batch_norm():
    """Verifies that `nn.BatchNorm` declares `single_batch_state` and updates shared running statistics."""
    bn = nn.BatchNorm(momentum=0.1)
    assert isinstance(bn, Node) and bn.parametric and bn.cyclic
    assert 'single_batch_state' in bn.tags

    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    model = batch(nn.Linear(4) >> bn).with_input(x).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    assert model.state.batch_norm.mean.shape == (4,)
    model, out = model(x)
    assert model.state.batch_norm.mean.shape == (4,)


def test_norm_running_stats_converge():
    """Verifies that shared running moments converge to the batch mean and variance under repeated steps."""
    x = jnp.array([[3.0], [5.0], [7.0]])
    cn = batch(nn.BatchNorm(momentum=0.1)).with_input(
        jnp.zeros_like(x)).parameterize().initialize()

    cn, _ = cn.scan(tile(x, 300))
    assert jnp.allclose(cn.state.mean, jnp.mean(x), atol=0.01)
    assert jnp.allclose(cn.state.var, jnp.var(x), atol=0.01)

    _, out = cn(x)
    assert jnp.allclose(jnp.mean(out), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out), 1.0, atol=0.05)


def test_linear_norm_pipe():
    """Verifies that `batch` binds the collective axis across `Linear >> BatchNorm` and converges running stats."""
    F = 4
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    assert isinstance(pipe, Node) and pipe.parametric and pipe.cyclic

    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0]])
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=5.0 * jnp.eye(F), b=10.0 * jnp.ones(F)),
        batch_norm=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    run = bound.initialize()

    run, outputs = run.scan(tile(x, 300))
    assert outputs.shape == (300, 3, F)
    expected_mean = jnp.mean(5.0 * x + 10.0, axis=0)
    assert jnp.allclose(run.state.batch_norm.mean, expected_mean, atol=0.1)

    _, out = run(x)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out, axis=0), 1.0, atol=0.05)


def test_linear_norm_eval_frozen_state():
    """Verifies that reusing a state-bound `PSNode` without advancing its successor evaluates deterministically."""
    F = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=2.0 * jnp.eye(F), b=jnp.ones(F)),
        batch_norm=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    run, _ = bound.initialize().scan(tile(x, 200))

    # Discarding the returned successor keeps the running statistics fixed across evaluation calls.
    _, out1 = run(x)
    _, out2 = run(x)
    _, out3 = run(x + 1.0)
    assert jnp.allclose(out1, out2)
    assert not jnp.allclose(out1, out3)


def test_linear_norm_dimension_change():
    """Verifies that `BatchNorm` parameter shapes infer their width from the preceding `Linear` output."""
    x = jnp.arange(20.0).reshape(5, 4)
    pipe = nn.Linear(3) >> nn.BatchNorm(momentum=0.1)
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=jnp.ones((4, 3)), b=jnp.zeros(3)),
        batch_norm=Struct(gamma=jnp.ones(3), beta=jnp.zeros(3)),
    ))
    run, _ = bound.initialize().scan(tile(x, 300))
    _, out = run(x)
    assert out.shape == (5, 3)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)


def test_train_linear_norm():
    """Verifies that `train_step(batch(pipe))` threads `BatchNorm` running statistics across optimization steps."""
    F = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    batched = batch(pipe).with_input(jnp.zeros_like(x))
    model = batched.bind(Struct(
        linear=Struct(w=jnp.eye(F), b=jnp.zeros(F)),
        batch_norm=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    )).initialize()
    trainer = train_step(model, mse, optax.adam(0.05))

    steps = 200
    final, (_, aux) = trainer.scan(input=tile(x, steps),
                                   target=jnp.zeros((steps, 2, F)))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]
    # Threaded running stats accumulate an EMA over all 200 steps rather than sitting near `momentum * batch_mean`.
    pre_norm = (x @ final.state.opt.params.model.linear.w
                + final.state.opt.params.model.linear.b)
    batch_mean = jnp.mean(pre_norm, axis=0)
    running = final.state.objective.model.batch_norm.mean
    assert jnp.linalg.norm(running) > 5 * jnp.linalg.norm(0.1 * batch_mean)


def test_nn_whiten():
    """Verifies that `nn.Whiten` computes cross-batch covariance moments and keeps one shared state copy."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    model = batch(nn.Whiten(momentum=0.1)).with_input(
        jnp.zeros_like(x)).initialize(input=x)
    assert model.state.mean.shape == (2,)
    assert model.state.cov.shape == (2, 2)

    model, out = model(x)
    assert out.shape == (2, 2)
    assert model.state.mean.shape == (2,)
    assert model.state.cov.shape == (2, 2)


# Named entries in `axis` reduce across enclosing mapped axes, integer entries pool leading sample axes at extent 1, and unlisted leading axes retain per-position statistics.

KEY = jax.random.PRNGKey(0)
MOMENTUM = 0.1


def test_batchnorm_flat_sample_keeps_per_feature_statistics() -> None:
    input = jax.random.normal(KEY, (4, 3)) * 2.0 + 1.0
    model = batch(nn.BatchNorm(MOMENTUM)).with_input(input).parameterize(
    ).initialize(input=input)
    assert model.state.mean.shape == (3,)

    successor, output = model.apply(input)
    assert output.shape == (4, 3)
    assert jnp.allclose(successor.state.mean,
                        MOMENTUM * jnp.mean(input, axis=0))
    assert jnp.allclose(successor.state.var,
                        (1 - MOMENTUM) + MOMENTUM * jnp.var(input, axis=0))


def test_batchnorm_sequence_sample_keeps_per_position_statistics() -> None:
    """Verifies that unpooled sequence samples keep separate running statistics at each sequence position."""
    input = jax.random.normal(KEY, (4, 5, 3))
    model = batch(nn.BatchNorm(MOMENTUM)).with_input(input).parameterize(
    ).initialize(input=input)
    assert model.state.mean.shape == (5, 3)

    successor, output = model.apply(input)
    assert output.shape == (4, 5, 3)
    assert jnp.allclose(successor.state.mean,
                        MOMENTUM * jnp.mean(input, axis=0))

    again, _ = successor.apply(input)
    assert again.state.mean.shape == (5, 3)


def test_batchnorm_pools_named_axes_and_generalizes_across_lengths() -> None:
    """Verifies that pooling a named stream axis collapses positions into length-independent feature statistics."""
    norm = nn.BatchNorm(MOMENTUM, axis=('batch', 'stream'))
    model = batch(batch(norm, axis='stream'), axis='batch')

    input = jax.random.normal(KEY, (4, 8, 3)) + 2.0
    bound = model.with_input(input).parameterize().initialize(input=input)
    assert bound.state.mean.shape == (3,)

    successor, output = bound.apply(input)
    assert output.shape == (4, 8, 3)
    assert jnp.allclose(successor.state.mean,
                        MOMENTUM * jnp.mean(input, axis=(0, 1)))

    short = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 3))
    rebound = model.with_input(short).parameterize().initialize(input=short)
    _, short_output = rebound.apply(short)
    assert short_output.shape == (4, 2, 3)


def test_batchnorm_int_axis_pools_the_sample_axis() -> None:
    """Verifies that `axis=('batch', 0)` pools the leading sample axis positionally at extent one."""
    batch_size = 4
    stream = 6
    features = 3
    input = jax.random.normal(KEY, (batch_size, stream, features)) + 1.0
    model = batch(nn.BatchNorm(MOMENTUM, axis=('batch', 0))).with_input(
        input).parameterize().initialize(input=input)
    assert model.state.mean.shape == (1, features)

    successor, output = model.apply(input)
    assert output.shape == (batch_size, stream, features)
    assert jnp.allclose(successor.state.mean,
                        MOMENTUM * jnp.mean(input, axis=(0, 1))[None, :])


def test_batchnorm_rejects_pooling_the_last_axis() -> None:
    """Verifies that `BatchNorm` rejects pooling the trailing feature axis."""
    model = batch(nn.BatchNorm(MOMENTUM, axis=('batch', -1)))
    input = jax.random.normal(KEY, (4, 3))
    with pytest.raises(ValueError, match='last axis'):
        model.with_input(input).parameterize().initialize(input=input)


def test_whiten_int_axis_pools_positionally() -> None:
    batch_size = 4
    stream = 5
    features = 3
    input = jax.random.normal(KEY, (batch_size, stream, features))
    model = batch(nn.Whiten(MOMENTUM, axis=('batch', 0))).with_input(
        input).initialize(input=input)
    assert model.state.cov.shape == (1, features, features)

    successor, _ = model.apply(input)
    centered = input - jnp.mean(input, axis=(0, 1), keepdims=True)
    pooled_cov = jnp.mean(
        jnp.einsum('bpi,bpj->bpij', centered, centered), axis=(0, 1))
    eye = jnp.eye(features)
    assert jnp.allclose(successor.state.cov[0],
                        (1 - MOMENTUM) * eye + MOMENTUM * pooled_cov,
                        atol=1e-5)


def test_batchnorm_unpooled_named_axis_fails_loudly() -> None:
    """Verifies that vmapping `BatchNorm` over an unpooled named axis raises an `out_axes` error."""
    model = batch(batch(nn.BatchNorm(MOMENTUM, axis='batch'), axis='stream'))
    input = jax.random.normal(KEY, (4, 8, 3))
    bound = model.with_input(input).parameterize().initialize(input=input)
    with pytest.raises(ValueError, match='out_axes'):
        bound.apply(input)


def test_streamed_rnn_normed_both_ways() -> None:
    """Verifies both positional per-step statistics and named cross-step pooling after `scanned(nn.RNN)`."""
    batch_size = 4
    stream = 6
    features = 3
    hidden = 5
    sequences = jax.random.normal(KEY, (batch_size, stream, features))
    recurrent = scanned(nn.RNN(hidden)).with_input(
        jnp.zeros((stream, features))).parameterize(rng=KEY)

    reference = batch(recurrent).initialize(input=sequences)
    _, pre_norm = reference.apply(sequences)
    assert pre_norm.shape == (batch_size, stream, hidden)

    positional_pipe = recurrent >> nn.BatchNorm(MOMENTUM)
    per_position = batch(positional_pipe).parameterize().initialize(
        input=sequences)
    assert per_position.state.batch_norm.mean.shape == (stream, hidden)
    successor, output = per_position.apply(sequences)
    assert output.shape == (batch_size, stream, hidden)
    assert jnp.allclose(successor.state.batch_norm.mean,
                        MOMENTUM * jnp.mean(pre_norm, axis=0), atol=1e-6)

    stream_norm = batch(nn.BatchNorm(MOMENTUM, axis=('batch', 'stream')), axis='stream')
    pooled_pipe = recurrent >> stream_norm
    pooled = batch(pooled_pipe).parameterize().initialize(input=sequences)
    assert pooled.state.batch.mean.shape == (hidden,)
    successor, output = pooled.apply(sequences)
    assert output.shape == (batch_size, stream, hidden)
    assert jnp.allclose(successor.state.batch.mean,
                        MOMENTUM * jnp.mean(pre_norm, axis=(0, 1)), atol=1e-6)


def test_whiten_sequence_sample_keeps_per_position_covariance() -> None:
    input = jax.random.normal(KEY, (4, 5, 3))
    model = batch(nn.Whiten(MOMENTUM)).with_input(input).parameterize(
    ).initialize(input=input)
    assert model.state.cov.shape == (5, 3, 3)

    successor, output = model.apply(input)
    assert output.shape == (4, 5, 3)
    centered = input - jnp.mean(input, axis=0)
    batch_cov = jnp.mean(
        jnp.einsum('bpi,bpj->bpij', centered, centered), axis=0)
    eye = jnp.broadcast_to(jnp.eye(3), (5, 3, 3))
    assert jnp.allclose(successor.state.cov,
                        (1 - MOMENTUM) * eye + MOMENTUM * batch_cov,
                        atol=1e-5)

    again, _ = successor.apply(input)
    assert again.state.cov.shape == (5, 3, 3)
