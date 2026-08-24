"""Learning a motor speed controller by backpropagating through the
simulated closed loop.

THE TASK — velocity setpoint tracking. A DC motor (a stateful simulation
component: winding current and rotor speed, saturated voltage input) must
follow step references in angular velocity. The controller driving it is
itself a stateful learned system: a committee of small RNNs behind a
running-stats normalizer, their voltage votes averaged into one command.
Training is plain gradient descent THROUGH the physics: unroll the closed
loop over the horizon, measure tracking error, differentiate the whole
rollout with respect to the controller weights (BPTT across physics,
actuator saturation, normalization and the committee).

Signal flow, one timestep of the closed loop:

    reference --(-)-- error --> norm --> committee(up >> stack(rnn) >> readout) --> mean
                 ^                                                      |
                 |                                              voltage |
                 +--------- measured velocity <---- motor <-------------+

And the training system is that loop, transformed:

    rollout = scanned(closed_loop(controller >> motor))  # ref seq -> velocity seq
    trainer = train_step(batch(rollout), mse, adam)   # fleet of setpoints, BPTT

test_assembly checks the composed structure (member namespace, input-resolved
state shapes, one-key param init); test_closed_loop_training checks that it
LEARNS: loss drops far, and the trained controller tracks a setpoint outside
its training set.

Self-contained: framework imports only; every domain component is
defined in this file. Features exercised together: cyclic simulation state, running-stats
normalization without mode flags, per-member per-layer recurrent state and
param draws under ensemble(n=) of stack(n=), constructor param init
(KeyStream) from one boundary key, input-resolved shape derivation (no shape
statics in the file), mixed bound/unbound pipe members, and
scan/batch/train_step closure.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax.binding import (split_aux)
from nodejax.transforms.train_step import learned_sgd
from nodejax import node, trained, Node, Leaf, serial, ensemble, reduce, stack, scan, scanned, batch, train_step, closed_loop
from nodejax.struct import Struct

DT = 0.05
T = 80                    # rollout horizon: 4 seconds
HIDDEN, MEMBERS, LAYERS = 4, 3, 2


# --- the simulation component: a saturated DC motor ---

@node
def Motor(dt: float, resistance: float=1.0, inductance: float=0.2, kt: float=1.0, ke: float=1.0,
              inertia: float=0.5, friction: float=0.5, v_max: float=6.0) -> Node:
    """DC motor with electrical and mechanical state plus actuator
    saturation: voltage command -> measured angular velocity."""
    def init(node):
        z = jnp.zeros_like(node.input)
        return Struct(current=z, omega=z)

    def apply(state, input):
        v = jnp.clip(input, -v_max, v_max)
        di = (v - resistance * state.current - ke * state.omega) / inductance
        domega = (kt * state.current - friction * state.omega) / inertia
        new = Struct(current=state.current + dt * di,
                     omega=state.omega + dt * domega)
        return new, new.omega

    return Leaf(apply, init=init)


# --- the controller: running-stats norm >> committee of RNNs >> mean ---

@node
def Norm(momentum: float=0.05, eps: float=1e-3) -> Node:
    """Streaming standardizer: running mean/var as cyclic state, shaped by
    the init input value. No mode flags, no shape statics."""
    def init(node):
        return Struct(mean=jnp.zeros_like(node.input), var=jnp.ones_like(node.input))

    def apply(state, input):
        new = Struct(mean=(1 - momentum) * state.mean + momentum * input,
                     var=(1 - momentum) * state.var + momentum * (input - state.mean) ** 2)
        return new, (input - state.mean) / jnp.sqrt(state.var + eps)

    return Leaf(apply, init=init)


@node
def Up(hidden: int) -> Node:
    """Input projection: scalar error -> (hidden,), so the signal threaded
    through the stacked RNN layers has a layer-invariant shape (a lax.scan
    carry may not change shape between layers)."""
    def param(rng):
        return Struct(win=0.5 * jax.random.normal(rng.next(), (hidden,)))

    def apply(param, input):
        return param.win * input

    return Leaf(apply, param=param)


@node
def RNN(hidden: int) -> Node:
    """Minimal recurrent cell over a (hidden,)-shaped signal. The param
    constructor is a plain callable — the tree(param) form: declaring rng,
    it receives a KeyStream and draws via rng.next(); explicit values could
    be passed instead, no key needed."""
    def param(rng):
        return Struct(
            wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
            wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
            b=jnp.zeros(hidden),
        )

    def init(param):
        return jnp.zeros(param.wh.shape[0])

    def apply(param, state, input):
        h = jnp.tanh(param.wx * input + param.wh @ state + param.b)
        return h, h

    return Leaf(apply, init=init, param=param)


@node
def Readout(hidden: int) -> Node:
    """Linear head: RNN hidden state -> scalar voltage command. Small init
    scale means the untrained controller commands near-zero voltage — a
    safe, stable starting policy."""
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (hidden,)),
                      b=jnp.zeros(()))

    def apply(param, input):
        return param.w @ input + param.b

    return Leaf(apply, param=param)


def Controller() -> Node:
    """The learned controller: tracking error in, voltage command out.
    The error is standardized against its running statistics, fed to a
    committee of MEMBERS independent DEEP recurrent controllers — each a
    stack of LAYERS RNN cells with per-member, per-layer params and hidden
    state — and their votes are averaged."""
    core = Up(HIDDEN) >> stack(RNN(HIDDEN), n=LAYERS) >> Readout(HIDDEN)
    return serial(norm=Norm(), committee=ensemble(core, n=MEMBERS),
                  reduce=reduce(jnp.mean))


def build() -> Node:
    """Assemble the system at both time scales: `loop` is the cyclic
    single-step closed loop (reference -> measured velocity), `rollout`
    its sequence-level scan (reference sequence -> velocity trajectory,
    controller and plant state internalized per episode)."""
    loop = closed_loop(Controller() >> Motor(DT))
    rollout = scanned(loop)
    return loop, rollout


def mse(pred: jax.Array, target: jax.Array):
    # a loss receives what the model emitted, aux included -> jax.Array: a gradient cell
    # inside the pipe reports its own loss on that channel, and this
    # objective has nothing to say about it
    pred, _ = split_aux(pred)
    return jnp.mean((pred - target) ** 2)


# --- tests ---

def test_assembly():
    """The full system assembles with flat named members, input-resolved
    state shapes, and rng-derived params: ONE key at parameterize splits
    down through the pipe, the committee, and every declared initializer."""
    loop, rollout = build()
    assert type(loop) is Node and loop.parametric and loop.cyclic
    assert type(rollout) is Node and not rollout.cyclic

    node = loop.parameterize(rng=jax.random.PRNGKey(0))
    # the loop holds the controlled pipe and its register, so the controller's
    # params live under the inner member
    wx = node.param.pipe.committee.stack_rnn.wx
    assert wx.shape == (MEMBERS, LAYERS, HIDDEN)   # stacked members x layers
    assert not jnp.allclose(wx[0], wx[1])          # independent member draws
    assert not jnp.allclose(wx[0, 0], wx[0, 1])    # independent layer draws

    # the rnn hidden-state ladder: each transform lifts the leaf state by
    # one axis — (H,) -> stack -> (L, H) -> ensemble (in the loop) -> (M, L, H)
    cell = RNN(HIDDEN).parameterize(rng=jax.random.PRNGKey(1))
    assert cell.init().shape == (HIDDEN,)
    deep = stack(RNN(HIDDEN), n=LAYERS).parameterize(rng=jax.random.PRNGKey(1))
    assert deep.init().shape == (LAYERS, HIDDEN)

    state = node.with_input(0.0).bind(node.param).init()

    assert state.pipe.committee.stack_rnn.shape == (MEMBERS, LAYERS, HIDDEN)
    assert state.pipe.norm.mean.shape == ()
    assert state.pipe.motor.current.shape == ()
    assert 'reduce' not in state.pipe              # stateless: no slot
    assert state.last.shape == ()


def test_closed_loop_training():
    """BPTT through the whole stack: physics, saturation, running-stats
    normalization, the RNN committee. Loss drops hard, and the trained
    controller generalizes to an unseen setpoint."""
    _, rollout = build()
    fleet = batch(rollout)   # several setpoints trained simultaneously

    model = fleet.parameterize(rng=jax.random.PRNGKey(0))

    wh = model.param.pipe.committee.stack_rnn.wh
    assert wh.shape == (MEMBERS, LAYERS, HIDDEN, HIDDEN)

    setpoints = jnp.array([0.5, 1.0, -0.5, 1.5])
    refs = setpoints[:, None] * jnp.ones(T)   # (4, T) step references

    # optimizer steps — each one unrolls a full T-timestep episode per
    # setpoint and applies one adam update on the same (stationary) task
    train_steps = 250

    def tile(x):
        return jnp.broadcast_to(x, (train_steps,) + x.shape)

    trainer = train_step(model.initialize(), mse, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(refs))

    assert jnp.all(jnp.isfinite(aux.loss))          # never destabilized the loop
    assert aux.loss[-1] < 0.35 * aux.loss[0]          # tracking improved substantially

    # generalization: an unseen setpoint, trained vs untrained params
    def track_mse(params, setpoint):
        ref = setpoint * jnp.ones(T)
        return mse(rollout.apply(params, ref), ref)

    assert track_mse(final.param, 0.8) < 0.5 * track_mse(model.param, 0.8)


def test_ttt_in_the_loop():
    """A fast-weights core inside the committee: each member's RNN adapts
    by one reconstruction gradient step per CONTROL step (test-time
    training), inside the closed loop, the ensemble, the scan, the batch
    and the outer trainer. The outer gradients flow through the inner
    ones, through the physics; the outer trainer meta-learns each
    member's initial weights and per-weight adaptation rates."""
    from nodejax import reconstruction_ttt

    inner_model = RNN(HIDDEN).parameterize(rng=jax.random.PRNGKey(1)).initialize()
    core = (Up(HIDDEN)
            >> reconstruction_ttt(train_step(inner_model, mse, learned_sgd(0.01)))
            >> Readout(HIDDEN))
    # the committee CONSTRUCTS its members from the core's def, one
    # independent draw each: the trainer binding overrides its
    # constructor, it does not delete it
    controller = serial(norm=Norm(), committee=ensemble(core.node, n=MEMBERS),
                        reduce=reduce(jnp.mean))
    fleet = batch(scanned(closed_loop(controller >> Motor(DT))))

    model = fleet.parameterize(rng=jax.random.PRNGKey(0))
    setpoints = jnp.array([0.5, 1.0, -0.5])
    refs = setpoints[:, None] * jnp.ones(T)
    steps = 150
    tile = lambda x: jnp.broadcast_to(x, (steps,) + x.shape)
    trainer = train_step(model.initialize(), mse, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(refs))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0], (aux.loss[0], aux.loss[-1])
