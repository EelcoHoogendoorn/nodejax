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

from nodejax import scan, scanned, PNode, Node, batch, train_step, nn
from nodejax.struct import Struct
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


from nodejax import tree_freeze

def test_batchnorm_unbatched_single_sample_eval():
    """Verify single-sample evaluation on unbatched BatchNorm (without batch transform) works when frozen."""
    bn = nn.BatchNorm(momentum=0.1)
    pipe = (nn.Linear(4) >> bn).with_input(jnp.zeros(4))
    bound_pipe = pipe.parameterize(rng=jax.random.PRNGKey(0))

    # freeze the binding whole: the running stats pin at their init
    frozen = tree_freeze(bound_pipe.initialize())
    _, sample_out = frozen(jnp.ones(4))
    assert sample_out.shape == (4,)


def test_nn_batch_norm():
    """Verify nn.BatchNorm in nn module has single_batch_state tag and updates running stats."""
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
    """A batch of three scalar samples: every element's collective sees
    the same batch moments, so the per-element stats agree and converge
    to the population's."""
    x = jnp.array([[3.0], [5.0], [7.0]])
    cn = batch(nn.BatchNorm(momentum=0.1)).with_input(
        jnp.zeros_like(x)).parameterize().initialize()

    cn, _ = cn.scan(tile(x, 300))
    assert jnp.allclose(cn.state.mean, jnp.mean(x), atol=0.01)
    assert jnp.allclose(cn.state.var, jnp.var(x), atol=0.01)

    # converged stats -> normalized output
    _, out = cn(x)
    assert jnp.allclose(jnp.mean(out), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out), 1.0, atol=0.05)


def test_linear_norm_pipe():
    """Linear >> Norm, per-sample; batch() binds the axis for the whole
    pipe, and the running stats converge to the batch statistics."""
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

    # scan over 300 identical batches to converge the running stats
    run, outputs = run.scan(tile(x, 300))
    assert outputs.shape == (300, 3, F)
    expected_mean = jnp.mean(5.0 * x + 10.0, axis=0)
    assert jnp.allclose(run.state.batch_norm.mean, expected_mean, atol=0.1)

    # after convergence the output is per-feature normalized
    _, out = run(x)
    assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=0.05)
    assert jnp.allclose(jnp.std(out, axis=0), 1.0, atol=0.05)


def test_linear_norm_eval_frozen_state():
    """Eval mode = reusing a frozen state; no flags anywhere."""
    F = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pipe = nn.Linear(F) >> nn.BatchNorm(momentum=0.1)
    bound = batch(pipe).with_input(jnp.zeros_like(x)).bind(Struct(
        linear=Struct(w=2.0 * jnp.eye(F), b=jnp.ones(F)),
        batch_norm=Struct(gamma=jnp.ones(F), beta=jnp.zeros(F)),
    ))
    run, _ = bound.initialize().scan(tile(x, 200))

    # eval reuses the run's state: successors deliberately dropped
    _, out1 = run(x)
    _, out2 = run(x)
    _, out3 = run(x + 1.0)
    assert jnp.allclose(out1, out2)          # frozen state -> deterministic
    assert not jnp.allclose(out1, out3)      # but still a function of input


def test_linear_norm_dimension_change():
    """Norm param shape couples to Linear out_features (4 -> 3)."""
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
    """Training a batchnormed model is just train_step(batch(pipe)): the
    running stats update during training because they are state, not
    because of a training-mode flag."""
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
    # the running stats THREADED through all 200 steps: they carry
    # full-scale history of the pre-norm activations (an EMA trailing
    # the still-moving weights). A state re-initialized every step
    # would sit at momentum * batch_mean — an order of magnitude
    # smaller and the decisive signature of broken threading.
    pre_norm = (x @ final.state.opt.params.model.linear.w
                + final.state.opt.params.model.linear.b)
    batch_mean = jnp.mean(pre_norm, axis=0)
    running = final.state.objective.model.batch_norm.mean
    assert jnp.linalg.norm(running) > 5 * jnp.linalg.norm(0.1 * batch_mean)


def test_nn_whiten():
    """Whiten is per-sample under the named axis, exactly as BatchNorm is:
    batch() binds the name, the moments are collectives over it, and the
    single_batch_state tag keeps one unbatched copy of the running stats."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    model = batch(nn.Whiten(momentum=0.1)).with_input(
        jnp.zeros_like(x)).initialize(input=x)
    assert model.state.mean.shape == (2,)
    assert model.state.cov.shape == (2, 2)

    model, out = model(x)
    assert out.shape == (2, 2)
    assert model.state.mean.shape == (2,)
    assert model.state.cov.shape == (2, 2)


# Moments reduce over exactly the listed axes: a name reaches across the
# enclosing map that binds it, an int selects a leading axis of the sample
# itself and stays in the statistics at extent one, and unlisted leading
# axes keep one running moment per position.

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
    """A sequence fed under a plain batch(): Cooijmans-style running moments,
    one per position, stable in shape across applications."""
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
    """Naming the sequence axis and listing it pools positions into flat
    per-feature statistics, which then apply at any length."""
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
    """axis=('batch', 0) pools the leading sample axis positionally: the
    same statistics as binding it to a name, kept at extent one."""
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
    """The last axis is the one the running statistics are laid out over,
    so pooling it positionally contradicts the node's own layout."""
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
    """Vmapping the norm over a named axis its moments do not pool cannot
    keep the shared running state consistent, and jax says so."""
    model = batch(batch(nn.BatchNorm(MOMENTUM, axis='batch'), axis='stream'))
    input = jax.random.normal(KEY, (4, 8, 3))
    bound = model.with_input(input).parameterize().initialize(input=input)
    with pytest.raises(ValueError, match='out_axes'):
        bound.apply(input)


def test_streamed_rnn_normed_both_ways() -> None:
    """The two readings of the stream axis at a norm downstream of a
    recurrence. scanned consumes the stream sequentially and restacks it, so
    at the norm it is plain data: left positional it flows into per-position
    running stats; bound to a name it is pooled out of them."""
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
    assert pooled.state.batch_batch_norm.mean.shape == (hidden,)
    successor, output = pooled.apply(sequences)
    assert output.shape == (batch_size, stream, hidden)
    assert jnp.allclose(successor.state.batch_batch_norm.mean,
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
