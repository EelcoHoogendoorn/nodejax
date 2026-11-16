"""Verifies test-time training (`ttt`, `supervised_ttt`, `reconstruction_ttt`, and `next_step_ttt`)."""

import jax
import jax.numpy as jnp
import pytest

from nodejax.transforms.learning import learned_sgd
from nodejax import (Node, train_step, Leaf, scan, scanned, stack, KeyStream,
                     supervised_ttt, next_step_ttt, reconstruction_ttt)
from nodejax.transforms.learning import ttt
from nodejax.struct import Struct
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


def Scale() -> Node:
    def param(scale=0.0) -> Struct:
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return Leaf(apply, param=param, name='g')


def sample(x: jax.Array):
    """Builds a self-supervised `(input, target)` pair where `target` equals `input`."""
    return Struct(input=jnp.asarray(x), target=jnp.asarray(x))


def test_ttt_adapts_toward_self_supervision():
    """Verifies reconstruction updates drive the carried gain parameter toward `1.0` across steps."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1))
    assert trainer.param.objective.model.scale == 0.0
    assert trainer.param.opt.model.scale == 0.1

    _, (outs, aux) = scan(trainer, record=True).apply(bundle=tile(sample(1.0), 50))
    scales = aux.state.opt.params.model.scale
    assert outs[0] == 0.0
    assert scales[0] > 0.0
    assert bool(jnp.all(jnp.diff(scales) >= 0))
    assert abs(float(scales[-1]) - 1.0) < 1e-2


def test_ttt_predicts_then_updates():
    """Verifies `train_step` predicts with incoming weights before applying the gradient update."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.5))
    successor, (out, _) = trainer.apply(bundle=sample(1.0))
    assert jnp.allclose(out, 0.0)
    # Evaluates -lr * d/ds (s - 1)^2 at s = 0.
    assert jnp.allclose(successor.state.opt.params.model.scale, 1.0)


def test_ttt_under_scan_resets_per_sequence():
    """Verifies `scan` starts each sequence from the bound initial weights while adapting within the sequence."""
    rollout = scan(train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1)))
    sequence = tile(sample(1.0), 20)
    _, (ys1, _) = rollout.apply(bundle=sequence)
    _, (ys2, _) = rollout.apply(bundle=sequence)
    assert jnp.allclose(ys1, ys2)
    assert ys1[-1] > ys1[0]


def test_ttt_marker_is_the_trainer_named():
    """Verifies `ttt` and `supervised_ttt` preserve predictions and `Aux` while renaming and validating the trainer."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1))
    sequence = tile(sample(1.0), 8)
    _, (preds_a, aux_a) = scan(trainer).apply(bundle=sequence)
    _, (preds_b, aux_b) = scan(supervised_ttt(trainer)).apply(bundle=sequence)
    assert jnp.allclose(preds_a, preds_b)
    assert jnp.allclose(aux_a.loss, aux_b.loss)
    assert ttt(trainer).name == f'ttt({trainer.name})'
    assert ttt(trainer).cyclic


def test_ttt_marker_refuses_a_bare_model():
    with pytest.raises(TypeError, match='product of train_step'):
        ttt(Scale())


def test_ttt_interior_to_a_pipe():
    """Verifies `reconstruction_ttt` adapts its carried weights when composed inside a `serial` pipeline."""
    layer = reconstruction_ttt(train_step(Scale().parameterize().initialize(),
                                          mse, learned_sgd(0.1)))
    model = Leaf(lambda input: 2.0 * input, name='pre') >> layer
    rollout = scan(model)

    _, (ys, aux) = rollout.apply(jnp.ones(20))
    losses = jax.tree.leaves(aux)[0]
    assert ys[0] == 0.0
    assert losses[-1] < losses[0]
    assert abs(ys[-1] - 2.0) < abs(ys[0] - 2.0)


def Lin() -> Node:
    """Builds a scalar linear `Leaf` with PRNG-initialized weight."""
    def param(rng: KeyStream) -> Struct:
        return Struct(w=0.5 + 0.1 * jax.random.normal(rng.next(), ()))

    def apply(param, input):
        return param.w * input

    return Leaf(apply, param=param, name='lin')


def RNNCell() -> Node:
    """Builds a scalar-input recurrent `Leaf` with hidden vector state."""
    def param(rng: KeyStream) -> Struct:
        return Struct(wx=0.5 * jax.random.normal(rng.next(), (4,)),
                      wh=0.3 * jax.random.normal(rng.next(), (4, 4)) / 2.0,
                      wo=0.5 * jax.random.normal(rng.next(), (4,)))

    def init(param):
        return jnp.zeros(param.wx.shape)

    def apply(param, state, input):
        h = jnp.tanh(param.wx * input + param.wh @ state)
        return h, param.wo @ h

    return Leaf(apply, init=init, param=param, name='rnn')


def test_both_deep_compositions_just_work():
    """Verifies `next_step_ttt` composes both outside `stack` (single head loss) and inside `stack` (per-layer losses)."""
    # Uses 200 online steps so deeper cells settle after upstream weights stabilize.
    data = (jnp.arange(200) % 2).astype(jnp.float32)

    # Single head loss backpropagated through the stacked RNN.
    deep = stack(RNNCell(), n=2).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    node = next_step_ttt(train_step(deep, mse, learned_sgd(0.1)))
    node = node.initialize(input=data[0])
    state, out = scan(node.pnode).apply(node.state, data)
    _, aux = out
    losses = aux.ttt_train_step_stack_rnn.loss
    assert losses[-1] < 0.3 * losses[1]
    fast = state.ttt_train_step_stack_rnn
    drift = jnp.max(jnp.abs(
        fast.opt.params.model.wx
        - node.param.ttt_train_step_stack_rnn.objective.model.wx), axis=1)
    assert drift.shape == (2,) and bool(jnp.all(drift > 0))
    assert fast.objective.model.shape == (2, 4)
    assert bool(jnp.any(fast.objective.model != 0))

    # Independent per-layer learners, constructed unbound so `stack` parameterizes each layer separately.
    cell = next_step_ttt(train_step(RNNCell(), mse, learned_sgd(0.1)))
    rollout = scanned(stack(cell, n=2)).parameterize(
        rng=jax.random.PRNGKey(0))
    ys, aux = rollout.apply(data)
    losses = jax.tree.leaves(aux)[0]
    assert ys.shape == (200,) and losses.shape == (200, 2)
    assert bool(jnp.all(jnp.mean(losses[-15:], axis=0) < 0.01))


def test_next_step_ttt_learns_the_shift():
    """Verifies `next_step_ttt` drives a stateless gain toward `-1.0` on an alternating sequence."""
    cell = next_step_ttt(train_step(Scale().parameterize().initialize(),
                                    mse, learned_sgd(0.2)))
    signal = jnp.tile(jnp.array([-1.0, 1.0]), 10)
    bound = cell.initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert float(final.state.ttt_train_step_g.opt.params.model.scale) < -0.9


def test_next_step_deep_compositions_just_work():
    """Verifies `next_step_ttt` primes cyclic registers across stacked linear layers under both head-loss and per-layer compositions."""
    signal = jnp.tile(jnp.array([-1.0, 1.0]), 30)

    deep = next_step_ttt(train_step(
        stack(Lin(), n=3).parameterize(rng=jax.random.PRNGKey(0)).initialize(),
        mse, learned_sgd(0.2)))
    bound = deep.initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert abs(float(jnp.prod(
        final.state.ttt_train_step_stack_lin.opt.params.model.w)) + 1.0) < 1e-2

    cell = next_step_ttt(train_step(Lin(), mse, learned_sgd(0.2)))
    layered = stack(cell, n=3)
    bound = layered.parameterize(rng=jax.random.PRNGKey(0)).initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert bool(jnp.all(
        final.state.ttt_train_step_lin.opt.params.model.w < -0.9))
