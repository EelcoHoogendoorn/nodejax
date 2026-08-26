"""Closure: the compositions that break on the per-corner implementation.

Transforms and pipes consume exactly what they produce, so transforms of
transforms, transforms of pipes, and pipes of transformed nodes all work.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, batch, ensemble, stack, scan, scanned, train_step, nn
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_batch_of_stack():
    """The readme's flagship: model = batch(stack(node))."""
    model = batch(stack(Gain(), n=2))
    node = model.bind(Struct(scale=jnp.array([2.0, 3.0])))  # two layers
    out = node.apply(jnp.array([1.0, 2.0, 4.0]))            # batch of three
    assert jnp.allclose(out, jnp.array([6.0, 12.0, 24.0]))


def test_ensemble_of_batch():
    model = ensemble(batch(Gain()), n=3)
    node = model.bind(Struct(scale=jnp.array([1.0, 2.0, 3.0])))  # three members
    out = node.apply(jnp.array([1.0, 2.0]))                      # batch of two, broadcast
    assert out.shape == (3, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))


def test_scan_of_stack():
    """Deep RNN, sequence level: scanned(stack(integrator))."""
    model = scanned(stack(Integrator(), n=2))
    node = model.parameterize()          # n says the depth; decay keeps its default
    outs = node.apply(jnp.array([1.0, 1.0, 1.0]))
    # layer 1 integrates the input; layer 2 integrates layer 1's output
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_ensemble_of_pipe():
    gain = Gain()
    pipe = gain >> gain
    model = ensemble(pipe, n=2)
    node = model.bind(Struct(
        gain=Struct(scale=jnp.array([1.0, 2.0])),
        gain_2=Struct(scale=jnp.array([10.0, 10.0])),
    ))
    out = node.apply(3.0)
    assert jnp.allclose(out, jnp.array([30.0, 60.0]))


def test_pipe_of_defs_and_bound_nodes():
    gain = Gain()
    # def-level composition, then bind with nested kwargs
    pipe = gain >> gain
    assert type(pipe) is Node and pipe.parametric and not pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0), gain_2=Struct(scale=3.0))
    assert bound.apply(1.0) == 6.0

    # bound-level composition gives the same result
    direct = gain.parameterize(scale=2.0) >> gain.parameterize(scale=3.0)
    assert direct.apply(1.0) == 6.0

    # >> stays flat across nesting
    triple = bound >> gain.parameterize(scale=10.0)
    assert set(triple.members.__keys__) == {'gain', 'gain_2', 'gain_3'}
    assert triple.apply(1.0) == 60.0


def test_pipe_mixed_cyclic():
    """Non-cyclic >> cyclic composes; pipe state is a Struct over all members."""
    pipe = Gain() >> Integrator()
    assert pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0))
    state = bound.init()
    state, out = bound.apply(state, 1.0)
    state, out = bound.apply(state, 1.0)
    assert jnp.allclose(out, 4.0)  # integrates 2.0 twice
    assert jnp.allclose(state.integrator, 4.0)
    assert 'gain' not in state  # stateless member: no slot, sparse throughout


def test_train_step_of_pipe():
    """Train a whole pipe: transform and composition compose."""
    gain = Gain()
    pipe = gain >> gain
    model = pipe.parameterize(gain=Struct(scale=jnp.array(1.0)),
                              gain_2=Struct(scale=jnp.array(1.0))).initialize()
    trainer = train_step(model, lambda pred, target: (pred - target) ** 2, optax.adam(0.05))
    final, (_, aux) = trainer.scan(input=jnp.full(1000, 2.0),
                                   target=jnp.full(1000, 12.0))

    product = final.state.opt.params.gain.scale * final.state.opt.params.gain_2.scale
    assert jnp.allclose(product, 6.0, atol=0.1)
    assert aux.loss[-1] < 1e-2


def test_train_step_of_stateful_pipe():
    """The contract form's free lunch: training a model WITH internal state
    (linear >> a smoothing filter), model state riding in trainer state."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pipe = (nn.Linear(3) >> nn.EMA(0.1)).with_input(x)
    model = pipe.parameterize(rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, lambda pred, target: jnp.mean((pred - target) ** 2), optax.adam(0.05))
    steps = 200
    final, (_, aux) = trainer.scan(input=jnp.tile(x, (steps, 1, 1)),
                                   target=jnp.zeros((steps, 2, 3)))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]
    # the model's own (EMA) state evolved inside the trainer state
    assert jnp.any(final.state.model.ema != 0.0)


def test_batch_of_pipe_with_state():
    """batch over a cyclic pipe: per-element pipe state, tiled by batch(pipe, n=...)."""
    pipe = Gain() >> Integrator()
    b = batch(pipe, n=2)
    bound = b.parameterize(gain=Struct(scale=1.0))
    state = bound.init()
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    assert jnp.allclose(out, jnp.array([2.0, 20.0]))
