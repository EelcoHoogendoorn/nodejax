"""Learning a motor speed controller by backpropagating through the
simulated closed loop.

THE TASK — velocity setpoint tracking. A DC motor (a stateful simulation
Node: winding current and rotor speed, saturated voltage input) must
follow step references in angular velocity. The controller driving it is
itself a stateful learned system: a committee of residual GRU stacks whose
voltage votes are averaged into one command.
Training is plain gradient descent THROUGH the physics: unroll the closed
loop over the horizon, measure tracking error, differentiate the whole
rollout with respect to the controller weights (BPTT across physics,
actuator saturation and the committee).

Signal flow, one timestep of the closed loop:

    reference --(-)-- error --> committee(linear >> stack(residual(gru)) >> projection) --> mean
                 ^                                                                      |
                 |                                                              voltage |
                 +---------------- measured velocity <---- motor <----------------------+

And the training system is that loop, transformed:

    rollout = scanned(closed_loop(controller >> motor))  # ref seq -> velocity seq
    trainer = train_step(batch(rollout), tracking_loss, adam)  # setpoints, BPTT

test_assembly checks the composed structure (member namespace, input-resolved
state shapes, one-key param init); test_closed_loop_training checks that it
LEARNS: loss drops far, and the trained controller tracks a setpoint outside
its training set.

Self-contained: framework imports only; every domain Node is defined in this file. Features
exercised together: cyclic simulation state, per-member per-layer recurrent state and param
draws under ensemble(n=) of stack(n=), constructor param init from one boundary key,
input-resolved shape derivation, mixed bound/unbound pipe members, and scan/batch/train_step
closure.
"""

import jax
import jax.numpy as jnp
import optax
from nodejax.binding import (split_aux)
from nodejax.transforms.train_step import learned_sgd
from nodejax import (
    Leaf, Node, batch, closed_loop, ensemble, nn, node, reduce, residual,
    scanned, stack, train_step, trained,
)
from nodejax.struct import Struct

DT = 0.05
T = 80                    # rollout horizon: 4 seconds
HIDDEN, MEMBERS, DEPTH = 4, 3, 2


# --- the simulation Node: a saturated DC motor ---

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


# --- the controller: committee of residual GRU stacks >> mean ---


def Controller() -> Node:
    """The learned controller: tracking error in, voltage command out.
    Each committee member expands the one-feature error, passes it through
    DEPTH residual GRUs, and projects it to a scalar voltage. Their votes are
    averaged."""
    core = (
        nn.Linear(HIDDEN)
        >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH)
        >> nn.Projection()
    )
    return ensemble(core, n=MEMBERS) >> reduce(jnp.mean)


def build() -> Node:
    """Assemble the system at both time scales: `loop` is the cyclic
    single-step closed loop (reference -> measured velocity), `rollout`
    its sequence-level scan (reference sequence -> velocity trajectory,
    controller and plant state internalized per episode)."""
    loop = closed_loop(Controller() >> Motor(DT))
    rollout = scanned(loop)
    return loop, rollout


def tracking_loss(pred: jax.Array, target: jax.Array):
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

    node = loop.with_input(jnp.zeros(1)).parameterize(rng=jax.random.PRNGKey(0))
    # the loop holds the controlled pipe and its register, so the controller's
    # params live under the inner member
    committee = node.param.pipe.ensemble_linear_stack_res_gru_projection
    assert committee.linear.w.shape == (MEMBERS, 1, HIDDEN)
    update_weight = committee.stack_res_gru.update_weight
    assert update_weight.shape == (MEMBERS, DEPTH, 2 * HIDDEN, HIDDEN)
    assert not jnp.allclose(update_weight[0], update_weight[1])
    assert not jnp.allclose(update_weight[0, 0], update_weight[0, 1])
    assert committee.projection.w.shape == (MEMBERS, HIDDEN)

    # Linear lifts the one-feature signal to hidden width; stack and ensemble
    # add the depth and member axes.
    cell = (
        nn.GRU(HIDDEN)
        .with_input(jnp.zeros(HIDDEN))
        .parameterize(rng=jax.random.PRNGKey(1))
    )
    assert cell.init().shape == (HIDDEN,)
    deep = (
        nn.Linear(HIDDEN) >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH)
    ).with_input(jnp.zeros(1)).parameterize(rng=jax.random.PRNGKey(1))
    assert deep.init().stack_res_gru.shape == (DEPTH, HIDDEN)

    state = node.init()

    committee_state = state.pipe.ensemble_linear_stack_res_gru_projection
    assert committee_state.stack_res_gru.shape == (MEMBERS, DEPTH, HIDDEN)
    assert state.pipe.motor.current.shape == ()
    assert 'reduce_mean' not in state.pipe
    assert state.last.shape == ()


def test_closed_loop_training():
    """BPTT through the physics, saturation, and GRU committee. Loss drops
    hard, and the trained controller generalizes to an unseen setpoint."""
    _, rollout = build()
    fleet = batch(rollout)   # several setpoints trained simultaneously

    setpoints = jnp.array([0.5, 1.0, -0.5, 1.5])
    refs = setpoints[:, None, None] * jnp.ones((1, T, 1))
    targets = refs[..., 0]
    model = fleet.with_input(refs).parameterize(rng=jax.random.PRNGKey(0))

    committee = model.param.pipe.ensemble_linear_stack_res_gru_projection
    update_weight = committee.stack_res_gru.update_weight
    assert update_weight.shape == (MEMBERS, DEPTH, 2 * HIDDEN, HIDDEN)

    # optimizer steps — each one unrolls a full T-timestep episode per
    # setpoint and applies one adam update on the same (stationary) task
    train_steps = 250

    def tile(x):
        return jnp.broadcast_to(x, (train_steps,) + x.shape)

    trainer = train_step(model.initialize(), tracking_loss, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(targets))

    assert jnp.all(jnp.isfinite(aux.loss))          # never destabilized the loop
    assert aux.loss[-1] < 0.35 * aux.loss[0]          # tracking improved substantially

    # generalization: an unseen setpoint, trained vs untrained params
    def track_mse(params, setpoint):
        ref = setpoint * jnp.ones((T, 1))
        return tracking_loss(rollout.apply(params, ref), ref[..., 0])

    assert track_mse(final.param, 0.8) < 0.5 * track_mse(model.param, 0.8)


def test_ttt_in_the_loop():
    """A fast-weights core inside the committee: each member's RNN adapts
    by one reconstruction gradient step per CONTROL step (test-time
    training), inside the closed loop, the ensemble, the scan, the batch
    and the outer trainer. The outer gradients flow through the inner
    ones, through the physics; the outer trainer meta-learns each
    member's initial weights and per-weight adaptation rates."""
    from nodejax import reconstruction_ttt

    inner_model = (
        nn.GRU(HIDDEN)
        .with_input(jnp.zeros(HIDDEN))
        .parameterize(rng=jax.random.PRNGKey(1))
        .initialize()
    )
    core = (
        nn.Linear(HIDDEN)
        >> reconstruction_ttt(
            train_step(inner_model, tracking_loss, learned_sgd(0.01)))
        >> nn.Projection()
    )
    # the committee CONSTRUCTS its members from the core's def, one
    # independent draw each: the trainer binding overrides its
    # constructor, it does not delete it
    controller = ensemble(core.node, n=MEMBERS) >> reduce(jnp.mean)
    fleet = batch(scanned(closed_loop(controller >> Motor(DT))))

    setpoints = jnp.array([0.5, 1.0, -0.5])
    refs = setpoints[:, None, None] * jnp.ones((1, T, 1))
    targets = refs[..., 0]
    model = fleet.with_input(refs).parameterize(rng=jax.random.PRNGKey(0))
    steps = 150
    tile = lambda x: jnp.broadcast_to(x, (steps,) + x.shape)
    trainer = train_step(model.initialize(), tracking_loss, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(targets))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0], (aux.loss[0], aux.loss[-1])
