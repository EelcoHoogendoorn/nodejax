"""Verifies algebraic closure across nested transforms and serial `>>` compositions."""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, batch, ensemble, stack, scan, scanned, train_step, nn
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def test_batch_of_stack():
    """Verifies `batch(stack(node))` over stacked layer parameters and a batched input."""
    model = batch(stack(Gain(), n=2))
    node = model.bind(Struct(scale=jnp.array([2.0, 3.0])))
    out = node.apply(jnp.array([1.0, 2.0, 4.0]))
    assert jnp.allclose(out, jnp.array([6.0, 12.0, 24.0]))


def test_ensemble_of_batch():
    model = ensemble(batch(Gain()), n=3)
    node = model.bind(Struct(scale=jnp.array([1.0, 2.0, 3.0])))
    out = node.apply(jnp.array([1.0, 2.0]))
    assert out.shape == (3, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))


def test_scan_of_stack():
    """Verifies `scanned(stack(Integrator(), n=2))` over a sequence input."""
    model = scanned(stack(Integrator(), n=2))
    node = model.parameterize()
    outs = node.apply(jnp.array([1.0, 1.0, 1.0]))
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
    pipe = gain >> gain
    assert type(pipe) is Node and pipe.parametric and not pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0), gain_2=Struct(scale=3.0))
    assert bound.apply(1.0) == 6.0

    direct = gain.parameterize(scale=2.0) >> gain.parameterize(scale=3.0)
    assert direct.apply(1.0) == 6.0

    # `>>` flattens nested serial pipes into one member namespace.
    triple = bound >> gain.parameterize(scale=10.0)
    assert set(triple.members.__keys__) == {'gain', 'gain_2', 'gain_3'}
    assert triple.apply(1.0) == 60.0


def test_pipe_mixed_cyclic():
    """Verifies that composing an acyclic member with a cyclic member keeps state sparse."""
    pipe = Gain() >> Integrator()
    assert pipe.cyclic
    bound = pipe.parameterize(gain=Struct(scale=2.0))
    state = bound.init()
    state, out = bound.apply(state, 1.0)
    state, out = bound.apply(state, 1.0)
    assert jnp.allclose(out, 4.0)
    assert jnp.allclose(state.integrator, 4.0)
    assert 'gain' not in state


def test_train_step_of_pipe():
    """Verifies `train_step` optimizing a serial pipe of parametric Nodes."""
    gain = Gain()
    pipe = gain >> gain
    model = pipe.parameterize(gain=Struct(scale=jnp.array(1.0)),
                              gain_2=Struct(scale=jnp.array(1.0))).initialize()
    trainer = train_step(model, lambda pred, target: (pred - target) ** 2, optax.adam(0.05))
    final, (_, aux) = trainer.scan(input=jnp.full(1000, 2.0),
                                   target=jnp.full(1000, 12.0))

    product = (final.state.opt.params.model.gain.scale
               * final.state.opt.params.model.gain_2.scale)
    assert jnp.allclose(product, 6.0, atol=0.1)
    assert aux.loss[-1] < 1e-2


def test_train_step_of_stateful_pipe():
    """Verifies `train_step` carrying internal model state (`Linear >> EMA`) inside trainer state."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pipe = (nn.Linear(3) >> nn.EMA(0.1)).with_input(x)
    model = pipe.parameterize(rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, lambda pred, target: jnp.mean((pred - target) ** 2), optax.adam(0.05))
    steps = 200
    final, (_, aux) = trainer.scan(input=jnp.tile(x, (steps, 1, 1)),
                                   target=jnp.zeros((steps, 2, 3)))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]
    assert jnp.any(final.state.objective.model.ema != 0.0)


def test_batch_of_pipe_with_state():
    """Verifies `batch(pipe, n=...)` tiling cyclic pipe state per batch element."""
    pipe = Gain() >> Integrator()
    b = batch(pipe, n=2)
    bound = b.parameterize(gain=Struct(scale=1.0))
    state = bound.init()
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    state, out = bound.apply(state, jnp.array([1.0, 10.0]))
    assert jnp.allclose(out, jnp.array([2.0, 20.0]))
