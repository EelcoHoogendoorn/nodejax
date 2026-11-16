"""Closed-loop motor velocity control trained by backpropagation through physics.

A stateful DC motor plant (`Motor`, tracking winding current and rotor angular
velocity under voltage saturation) follows step velocity references driven by a
committee of residual GRU stacks (`Controller`). Differentiating the unrolled
closed loop updates controller weights across plant dynamics, actuator clipping,
and recurrent hidden states:

    reference --(-)-- error --> ensemble(linear >> stack(residual(gru)) >> projection) --> mean
                 ^                                                                          |
                 |                                                                  voltage |
                 +-------------------- measured velocity <---- motor <----------------------+

Composition across single-step feedback, episode rollouts, setpoints, and training:

    rollout = scanned(closed_loop(controller >> motor))
    trainer = train_step(batch(rollout), tracking_loss, adam)

`test_assembly` verifies member hierarchy, shape propagation, and PRNG splitting
across `ensemble` and `stack`. `test_closed_loop_training` verifies closed-loop
parameter updates and rollout evaluation on an unseen setpoint, and
`test_ttt_in_the_loop` nests per-timestep `reconstruction_ttt` adaptation inside
each committee member.
"""

import jax
import jax.numpy as jnp
import optax
from nodejax.transforms.learning import learned_sgd
from nodejax import (
    Leaf, Node, PNode, batch, closed_loop, drop_aux, ensemble, nn, node, reduce,
    residual, scanned, stack, train_step, trained,
)
from nodejax.struct import Struct

DT = 0.05
T = 12
HIDDEN, MEMBERS, DEPTH = 4, 3, 2


@node
def Motor(dt: float, resistance: float = 1.0, inductance: float = 0.2,
          kt: float = 1.0, ke: float = 1.0, inertia: float = 0.5,
          friction: float = 0.5, v_max: float = 6.0) -> PNode:
    """Simulates a voltage-saturated DC motor, advancing current and angular velocity each step."""
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


def Controller() -> Node:
    """Builds an ensemble of residual GRU stacks whose scalar voltage commands are averaged."""
    core = (
        nn.Linear(HIDDEN)
        >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH)
        >> nn.Projection()
    )
    return ensemble(core, n=MEMBERS) >> reduce(jnp.mean)


def build() -> tuple[Node, Node]:
    """Constructs both the single-step `closed_loop` Node and its episode-level `scanned` rollout."""
    loop = closed_loop(Controller() >> Motor(DT))
    rollout = scanned(loop)
    return loop, rollout


def tracking_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean squared velocity tracking error."""
    return jnp.mean((pred - target) ** 2)


def test_assembly() -> None:
    """Verifies member namespaces, input-derived state shapes, and single-key parameterization across `ensemble` and `stack`."""
    loop, rollout = build()
    assert type(loop) is Node and loop.parametric and loop.cyclic
    assert type(rollout) is Node and not rollout.cyclic

    node = loop.with_input(jnp.zeros(1)).parameterize(rng=jax.random.PRNGKey(0))
    # `closed_loop` stores the controlled chain under `pipe` alongside its feedback register.
    committee = node.param.pipe.ensemble
    assert committee.linear.w.shape == (MEMBERS, 1, HIDDEN)
    update_weight = committee.stack.update_weight
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
    assert deep.init().stack.shape == (DEPTH, HIDDEN)

    state = node.init()

    committee_state = state.pipe.ensemble
    assert committee_state.stack.shape == (MEMBERS, DEPTH, HIDDEN)
    assert state.pipe.motor.current.shape == ()
    assert 'reduce' not in state.pipe
    assert state.last.shape == ()


def test_closed_loop_training() -> None:
    """Verifies closed-loop parameter updates through plant dynamics and rollout evaluation on an unseen setpoint."""
    _, rollout = build()
    batched_rollout = batch(rollout)

    setpoints = jnp.array([0.5, 1.0, -0.5, 1.5])
    refs = setpoints[:, None, None] * jnp.ones((1, T, 1))
    targets = refs[..., 0]
    model = batched_rollout.with_input(refs).parameterize(rng=jax.random.PRNGKey(0))

    committee = model.param.pipe.ensemble
    update_weight = committee.stack.update_weight
    assert update_weight.shape == (MEMBERS, DEPTH, 2 * HIDDEN, HIDDEN)

    # Each optimizer step unrolls a `T`-timestep episode across all setpoints.
    train_steps = 3

    def tile(x):
        return jnp.broadcast_to(x, (train_steps,) + x.shape)

    trainer = train_step(model.initialize(), tracking_loss, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(targets))

    assert aux.loss.shape == (train_steps,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, model.param))

    unseen = drop_aux(rollout.apply(final.param, 0.8 * jnp.ones((T, 1))))
    assert unseen.shape == (T,)
    assert jnp.all(jnp.isfinite(unseen))


def test_ttt_in_the_loop() -> None:
    """Verifies outer meta-training of initial weights and `learned_sgd` rates around per-step `reconstruction_ttt` inside the closed loop."""
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
    # Passing `core.node` to `ensemble` draws independent parameter initializations per committee member.
    controller = ensemble(core.node, n=MEMBERS) >> reduce(jnp.mean)
    batched_rollout = batch(scanned(closed_loop(controller >> Motor(DT))))

    setpoints = jnp.array([0.5, 1.0, -0.5])
    refs = setpoints[:, None, None] * jnp.ones((1, T, 1))
    targets = refs[..., 0]
    model = batched_rollout.with_input(refs).parameterize(rng=jax.random.PRNGKey(0))
    steps = 3
    tile = lambda x: jnp.broadcast_to(x, (steps,) + x.shape)
    trainer = train_step(model.initialize(), tracking_loss, optax.adam(0.02))
    final, aux = trained(trainer).apply(input=tile(refs), target=tile(targets))

    assert aux.loss.shape == (steps,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, model.param))
