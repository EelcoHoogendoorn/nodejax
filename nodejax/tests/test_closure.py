"""Closure: the compositions that break on the per-corner implementation.

Transforms and pipes consume exactly what they produce, so transforms of
transforms, transforms of pipes, and pipes of transformed nodes all work.
"""

import jax.numpy as jnp
import optax

from nodejax import NodeDef, batch, ensemble, stack, scan, train_step
from nodejax.struct import Struct
from nodejax.examples import gain_def, integrator_def, Linear, Ema


def test_batch_of_stack():
    """The readme's flagship: model = batch(stack(node))."""
    model = batch(stack(gain_def()))
    node = model.parameterize(scale=jnp.array([2.0, 3.0]))  # two layers
    out = node.apply(jnp.array([1.0, 2.0, 4.0]))            # batch of three
    assert jnp.allclose(out, jnp.array([6.0, 12.0, 24.0]))


def test_ensemble_of_batch():
    model = ensemble(batch(gain_def()))
    node = model.parameterize(scale=jnp.array([1.0, 2.0, 3.0]))  # three members
    out = node.apply(jnp.array([1.0, 2.0]))                      # batch of two, broadcast
    assert out.shape == (3, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))


def test_scan_of_stack():
    """Deep RNN, sequence level: scan(stack(integrator))."""
    model = scan(stack(integrator_def()))
    node = model.parameterize(gain=jnp.array([1.0, 1.0]))  # two integrator layers
    outs = node.apply(jnp.array([1.0, 1.0, 1.0]))
    # layer 1 integrates the input; layer 2 integrates layer 1's output
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_ensemble_of_pipe():
    Gain = gain_def()
    pipe = Gain >> Gain
    model = ensemble(pipe)
    node = model.parameterize(
        gain=Struct(scale=jnp.array([1.0, 2.0])),
        gain_2=Struct(scale=jnp.array([10.0, 10.0])),
    )
    out = node.apply(3.0)
    assert jnp.allclose(out, jnp.array([30.0, 60.0]))


def test_pipe_of_defs_and_bound_nodes():
    Gain = gain_def()
    # def-level composition, then bind with nested kwargs
    pipe = Gain >> Gain
    assert isinstance(pipe, NodeDef) and pipe.parametric and not pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0), gain_2=Struct(scale=3.0))
    assert bound.apply(1.0) == 6.0

    # bound-level composition gives the same result
    direct = Gain.parameterize(scale=2.0) >> Gain.parameterize(scale=3.0)
    assert direct.apply(1.0) == 6.0

    # >> stays flat across nesting
    triple = bound >> Gain.parameterize(scale=10.0)
    assert set(triple.ndef.members) == {'gain', 'gain_2', 'gain_3'}
    assert triple.apply(1.0) == 60.0


def test_pipe_mixed_cyclic():
    """Non-cyclic >> cyclic composes; pipe state is a Struct over all members."""
    pipe = gain_def() >> integrator_def()
    assert pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0), integrator=Struct(gain=1.0))
    state = bound.init()
    state, out = bound.apply(state, 1.0)
    state, out = bound.apply(state, 1.0)
    assert jnp.allclose(out, 4.0)  # integrates 2.0 twice
    assert jnp.allclose(state.integrator, 4.0)
    assert state.gain == ()  # trivial member state composes without special cases


def test_train_step_of_pipe():
    """Train a whole pipe: transform and composition compose."""
    Gain = gain_def()
    pipe = Gain >> Gain
    trainer = train_step(pipe, lambda pred, target: (pred - target) ** 2, optax.adam(0.05))

    model = pipe.parameterize(gain=Struct(scale=jnp.array(1.0)), gain_2=Struct(scale=jnp.array(1.0)))
    state = trainer.init(model=model.param)
    inputs = Struct(input=jnp.full(1000, 2.0), target=jnp.full(1000, 12.0))
    final, losses = trainer.scan(state, inputs)

    product = final.model.gain.scale * final.model.gain_2.scale
    assert jnp.allclose(product, 6.0, atol=0.1)
    assert losses[-1] < 1e-2


def test_train_step_of_stateful_pipe():
    """The contract form's free lunch: training a model WITH internal state
    (linear >> running-mean normalizer), model state riding in trainer state."""
    pipe = Linear(3, 3) >> Ema(0.1)
    trainer = train_step(pipe, lambda pred, target: jnp.mean((pred - target) ** 2), optax.adam(0.05))

    model = pipe.parameterize(
        linear3x3=Struct(weight=jnp.eye(3) * 5.0, bias=jnp.ones(3)),
        ema=Struct(gamma=jnp.ones(3), beta=jnp.zeros(3)),
    )
    state = trainer.init(model=model.param)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    steps = 200
    inputs = Struct(
        input=jnp.tile(x, (steps, 1, 1)),
        target=jnp.zeros((steps, 2, 3)),
    )
    final, losses = trainer.scan(state, inputs)

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]
    # the model's own (EMA) state evolved inside the trainer state
    assert final.inner.ema != 0.0


def test_batch_of_pipe_with_state():
    """batch over a cyclic pipe: per-element pipe state, tiled by batch(pipe, n=...)."""
    pipe = gain_def() >> integrator_def()
    b = batch(pipe, n=2)
    bound = b.parameterize(gain=Struct(scale=1.0), integrator=Struct(gain=1.0))
    state = bound.init()
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    assert jnp.allclose(out, jnp.array([2.0, 20.0]))
