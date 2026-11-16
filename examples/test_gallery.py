"""Algebraic transform composition across filtering, system identification, ensembles, and control.

Covers commutativity of `scanned` and `stack` on linear time-invariant filters, backpropagation
through time via `train_step(scanned(...))`, parallel ensemble optimization via
`train_step(ensemble(...))`, and differentiable proportional-derivative controller tuning through
simulated spring-damper physics via `train_step(scanned(feedback(...)))`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, PNode, Leaf, ensemble, stack, scan, scanned, train_step
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator, PD, feedback
from nodejax import tile
from examples.util import mse


def Plant(dt: float = 0.01, spring_k: float = 1.0, damping_c: float = 0.1) -> PNode:
    """Integrates one Euler step of a damped harmonic oscillator mapping force `input` to position."""
    def init(node) -> Struct:
        return Struct(pos=jnp.zeros_like(node.input), vel=jnp.zeros_like(node.input))

    def apply(state, input) -> tuple[Struct, jax.Array]:
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos

    return Leaf(apply, init=init, name='plant')


def test_stack_scan_commute() -> None:
    """Verifies that `scanned(stack(layer, n=3))` and `stack(scanned(layer), n=3)` agree on LTI filters."""
    gains = jnp.array([1.0, 0.5, 2.0])
    xs = jnp.array([1.0, -1.0, 2.0, 0.5])

    layer = Gain() >> Integrator()
    scaled = Struct(gain=Struct(scale=gains), integrator=Struct(decay=jnp.zeros(3)))
    per_step = scanned(stack(layer, n=3)).bind(scaled)
    per_layer = stack(scanned(layer), n=3).bind(scaled)

    assert jnp.allclose(per_step.apply(xs), per_layer.apply(xs))


def test_bptt_system_identification() -> None:
    """Recovers the decay constant of a recurrent leaky integrator with `train_step(scanned(Leaky()))`."""
    def Leaky() -> Node:
        def param(decay: jax.Array) -> Struct:
            return Struct(decay=jnp.asarray(decay))

        def init(param) -> float:
            return 0.0

        def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
            new = param.decay * state + input
            return new, new

        return Leaf(apply, param=param, init=init, name='leaky')

    seq = scanned(Leaky())
    xs = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    target_ys = seq.parameterize(decay=jnp.array(0.7)).apply(xs)

    trainer = train_step(seq.parameterize(decay=jnp.array(0.1)).initialize(),
                         mse, optax.adam(0.05))
    steps = 300
    final, (_, aux) = trainer.scan(input=tile(xs, steps),
                                   target=tile(target_ys, steps))

    assert jnp.allclose(final.state.opt.params.model.decay, 0.7, atol=0.01)
    assert aux.loss[-1] < 1e-6


def test_deep_ensemble_training() -> None:
    """Trains three independent `Gain` members jointly with `train_step(ensemble(Gain(), n=3))`."""
    ens = ensemble(Gain(), n=3).bind(Struct(scale=jnp.array([-1.0, 0.0, 5.0])))
    trainer = train_step(ens.initialize(), mse, optax.sgd(0.05))
    steps = 400
    final, (_, aux) = trainer.scan(input=tile(jnp.array(2.0), steps),
                                   target=tile(jnp.array(6.0), steps))

    assert jnp.allclose(final.state.opt.params.model.scale, jnp.full(3, 3.0), atol=0.01)


def test_gradient_pid_tuning() -> None:
    """Tunes `PD` controller gains by differentiating through an unrolled closed-loop plant rollout."""
    dt = 0.1

    loop = feedback(PD(dt) >> Plant(dt, 1.0, 0.3).node)
    rollout = scanned(loop)

    T = 40
    reference = jnp.ones(T)
    trainer = train_step(rollout.parameterize(
        pipe=Struct(pd=Struct(kp=jnp.array(1.0), kd=jnp.array(0.0)))).initialize(),
        mse, optax.adam(0.05))

    steps = 300
    final, (_, aux) = trainer.scan(input=tile(reference, steps),
                                   target=tile(reference, steps))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0]
    assert final.state.opt.params.model.pipe.pd.kp > 1.0
    assert final.state.opt.params.model.pipe.pd.kd > 0.0


def test_feedback_mimo() -> None:
    """Verifies steady-state step tracking of a two-axis `feedback` loop against `kp / (kp + spring_k)`."""
    dt, spring_k = 0.05, 1.0

    # `output_spec` sizes the feedback output register for the two plant axes.
    loop = feedback(PD(dt) >> Plant(dt, spring_k, 0.3).node, output_spec=jnp.zeros(2))
    kp = jnp.array([4.0, 9.0])
    bound = scanned(loop).parameterize(pipe=Struct(pd=Struct(kp=kp, kd=jnp.array(1.0))))

    T = 600
    trajectory = bound.apply(jnp.ones((T, 2)))

    assert trajectory.shape == (T, 2)
    assert jnp.allclose(trajectory[-1], kp / (kp + spring_k), atol=0.02)
