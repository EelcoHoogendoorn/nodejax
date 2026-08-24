"""A compositionality gallery: a transform-algebra identity
(scan . stack == stack . scan on linear time-invariant nodes), system
identification via BPTT (train_step(scanned(rnn))), deep ensembles in one line
(train_step(ensemble(model))), and gradient-based PID tuning through
simulated plant physics (train_step(scanned(feedback(pd >> plant)))).
"""

import jax.numpy as jnp
import optax

from nodejax import Node, Leaf, ensemble, stack, scan, scanned, train_step
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator, PD, feedback
from nodejax import tile
from nodejax.examples.util import mse


def Plant(dt: float=0.01, spring_k: float=1.0, damping_c: float=0.1) -> Node:
    def init(node):
        return Struct(pos=jnp.zeros_like(node.input), vel=jnp.zeros_like(node.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return Leaf(apply, init=init, name='plant')


def test_stack_scan_commute():
    """A theorem of the transform algebra: scanned(stack(x)) and stack(scanned(x))
    are different programs — step through all layers per timestep vs filter
    the whole signal layer by layer — but for linear time-invariant nodes
    they are the same operation (cascaded filters commute with time)."""
    gains = jnp.array([1.0, 0.5, 2.0])
    xs = jnp.array([1.0, -1.0, 2.0, 0.5])

    layer = Gain() >> Integrator()          # the gain is its own block
    scaled = Struct(gain=Struct(scale=gains), integrator=Struct(decay=jnp.zeros(3)))
    per_step = scanned(stack(layer, n=3)).bind(scaled)
    per_layer = stack(scanned(layer), n=3).bind(scaled)

    assert jnp.allclose(per_step.apply(xs), per_layer.apply(xs))


def test_bptt_system_identification():
    """BPTT as composition: train_step(scanned(rnn)) trains a recurrent node at
    the sequence level, gradients flowing through the unrolled state loop —
    recovering the true decay constant of a leaky integrator exactly."""
    def Leaky():
        def param(decay):
            return Struct(decay=jnp.asarray(decay))
        def init(param):
            return 0.0
        def apply(param, state, input):
            new = param.decay * state + input
            return new, new
        return Leaf(apply, param=param, init=init, name='leaky')

    seq = scanned(Leaky())  # PCN -> PN: maps input sequence -> output sequence
    xs = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    target_ys = seq.parameterize(decay=jnp.array(0.7)).apply(xs)

    trainer = train_step(seq.parameterize(decay=jnp.array(0.1)).initialize(),
                         mse, optax.adam(0.05))
    steps = 300
    final, (_, aux) = trainer.scan(input=tile(xs, steps),
                                   target=tile(target_ys, steps))

    assert jnp.allclose(final.state.opt.params.decay, 0.7, atol=0.01)
    assert aux.loss[-1] < 1e-6


def test_deep_ensemble_training():
    """Deep ensembles in one line: train_step(ensemble(model)) trains N
    independently-initialized members jointly; every member receives its own
    gradients and converges to the optimum."""
    ens = ensemble(Gain(), n=3).bind(Struct(scale=jnp.array([-1.0, 0.0, 5.0])))
    trainer = train_step(ens.initialize(), mse, optax.sgd(0.05))
    steps = 400
    final, (_, aux) = trainer.scan(input=tile(jnp.array(2.0), steps),
                                   target=tile(jnp.array(6.0), steps))

    assert jnp.allclose(final.state.opt.params.scale, jnp.full(3, 3.0), atol=0.01)


def test_gradient_pid_tuning():
    """Control and ML in one algebra — the library's raison d'etre.

    A PD controller is tuned by gradient descent THROUGH the simulated plant
    physics: train_step(scanned(feedback(pd >> plant))). The whole closed-loop
    rollout is a pure differentiable function of the gains. Note: no shape
    arguments anywhere — all state derives from the reference sequence via
    init input values."""
    dt = 0.1

    loop = feedback(PD(dt) >> Plant(dt, 1.0, 0.3).node)  # reference -> position
    rollout = scanned(loop)                                          # reference sequence -> trajectory

    T = 40
    reference = jnp.ones(T)  # step response
    trainer = train_step(rollout.parameterize(
        pipe=Struct(pd=Struct(kp=jnp.array(1.0), kd=jnp.array(0.0)))).initialize(),
        mse, optax.adam(0.05))

    steps = 300
    final, (_, aux) = trainer.scan(input=tile(reference, steps),
                                   target=tile(reference, steps))

    assert jnp.all(jnp.isfinite(aux.loss))          # training never destabilized the loop
    assert aux.loss[-1] < 0.3 * aux.loss[0]           # tracking substantially improved
    assert final.state.opt.params.pipe.pd.kp > 1.0    # stiffer P gain to fight spring droop
    assert final.state.opt.params.pipe.pd.kd > 0.0    # learned to add damping


def test_feedback_mimo():
    """The closed loop is not scalar-specific: a two-axis plant with per-axis
    gains and NO shape arguments — every piece of loop state (previous
    error, plant pos/vel, feedback register) derives from the reference sequence
    via init input values. Steady state matches the analytic P-control
    droop kp/(kp + spring_k) per axis."""
    dt, spring_k = 0.05, 1.0

    # two axes, so the register starts from a shaped zero
    loop = feedback(PD(dt) >> Plant(dt, spring_k, 0.3).node, output_spec=jnp.zeros(2))
    kp = jnp.array([4.0, 9.0])
    bound = scanned(loop).parameterize(pipe=Struct(pd=Struct(kp=kp, kd=jnp.array(1.0))))

    T = 600
    trajectory = bound.apply(jnp.ones((T, 2)))  # track a unit step on both axes

    assert trajectory.shape == (T, 2)
    assert jnp.allclose(trajectory[-1], kp / (kp + spring_k), atol=0.02)
