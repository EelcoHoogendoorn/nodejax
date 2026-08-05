"""A compositionality gallery: a transform-algebra identity
(scan . stack == stack . scan on linear time-invariant nodes), system
identification via BPTT (train_step(scan(rnn))), deep ensembles in one line
(train_step(ensemble(model))), and gradient-based PID tuning through
simulated plant physics (train_step(scan(feedback(pd >> plant)))).
"""

import jax.numpy as jnp
import optax

from nodejax import node_def, ensemble, stack, scan, train_step
from nodejax.struct import Struct
from nodejax.examples import (gain_def, integrator_def, mse, tile,
                                    feedback, pd_def, plant_node)


def test_stack_scan_commute():
    """A theorem of the transform algebra: scan(stack(x)) and stack(scan(x))
    are different programs — step through all layers per timestep vs filter
    the whole signal layer by layer — but for linear time-invariant nodes
    they are the same operation (cascaded filters commute with time)."""
    gains = jnp.array([1.0, 0.5, 2.0])
    xs = jnp.array([1.0, -1.0, 2.0, 0.5])

    per_step = scan(stack(integrator_def())).parameterize(gain=gains)
    per_layer = stack(scan(integrator_def())).parameterize(gain=gains)

    assert jnp.allclose(per_step.apply(xs), per_layer.apply(xs))


def test_bptt_system_identification():
    """BPTT as composition: train_step(scan(rnn)) trains a recurrent node at
    the sequence level, gradients flowing through the unrolled state loop —
    recovering the true decay constant of a leaky integrator exactly."""
    def leaky_def():
        def param(decay):
            return Struct(decay=jnp.asarray(decay))
        def init(param):
            return 0.0
        def apply(param, state, input):
            new = param.decay * state + input
            return new, new
        return node_def(apply, param=param, init=init, name='leaky')

    seq = scan(leaky_def())  # PCN -> PN: maps input sequence -> output sequence
    xs = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    target_ys = seq.parameterize(decay=jnp.array(0.7)).apply(xs)

    trainer = train_step(seq, mse, optax.adam(0.05))
    state = trainer.init(model=seq.parameterize(decay=jnp.array(0.1)).param)
    steps = 300
    inputs = Struct(input=tile(xs, steps), target=tile(target_ys, steps))
    final, losses = trainer.scan(state, inputs)

    assert jnp.allclose(final.model.decay, 0.7, atol=0.01)
    assert losses[-1] < 1e-6


def test_deep_ensemble_training():
    """Deep ensembles in one line: train_step(ensemble(model)) trains N
    independently-initialized members jointly; every member receives its own
    gradients and converges to the optimum."""
    ens = ensemble(gain_def())
    trainer = train_step(ens, mse, optax.sgd(0.05))

    state = trainer.init(model=ens.parameterize(scale=jnp.array([-1.0, 0.0, 5.0])).param)
    steps = 400
    inputs = Struct(input=tile(jnp.array(2.0), steps),
                    target=tile(jnp.array(6.0), steps))
    final, losses = trainer.scan(state, inputs)

    assert jnp.allclose(final.model.scale, jnp.full(3, 3.0), atol=0.01)


def test_gradient_pid_tuning():
    """Control and ML in one algebra — the library's raison d'etre.

    A PD controller is tuned by gradient descent THROUGH the simulated plant
    physics: train_step(scan(feedback(pd >> plant))). The whole closed-loop
    rollout is a pure differentiable function of the gains. Note: no shape
    arguments anywhere — all state derives from the reference sequence via
    init input values."""
    dt = 0.1

    loop = feedback(pd_def(dt) >> plant_node(dt, 1.0, 0.3).ndef)  # reference -> position
    rollout = scan(loop)                                          # reference sequence -> trajectory

    T = 40
    reference = jnp.ones(T)  # step response
    trainer = train_step(rollout, mse, optax.adam(0.05))
    state = trainer.init(model=rollout.parameterize(
        pd=Struct(kp=jnp.array(1.0), kd=jnp.array(0.0))).param)

    steps = 300
    inputs = Struct(input=tile(reference, steps), target=tile(reference, steps))
    final, losses = trainer.scan(state, inputs)

    assert jnp.all(jnp.isfinite(losses))          # training never destabilized the loop
    assert losses[-1] < 0.3 * losses[0]           # tracking substantially improved
    assert final.model.pd.kp > 1.0                # stiffer P gain to fight spring droop
    assert final.model.pd.kd > 0.0                # learned to add damping


def test_feedback_mimo():
    """The closed loop is not scalar-specific: a two-axis plant with per-axis
    gains and NO shape arguments — every piece of loop state (previous
    error, plant pos/vel, feedback seed) derives from the reference sequence
    via init input values. Steady state matches the analytic P-control
    droop kp/(kp + spring_k) per axis."""
    dt, spring_k = 0.05, 1.0

    loop = feedback(pd_def(dt) >> plant_node(dt, spring_k, 0.3).ndef)
    kp = jnp.array([4.0, 9.0])
    bound = scan(loop).parameterize(pd=Struct(kp=kp, kd=jnp.array(1.0)))

    T = 600
    trajectory = bound.apply(jnp.ones((T, 2)))  # track a unit step on both axes

    assert trajectory.shape == (T, 2)
    assert jnp.allclose(trajectory[-1], kp / (kp + spring_k), atol=0.02)
