"""Meta-learned motor controller auto-tuning across domain-randomized plants.

A recurrent velocity controller ships into installations whose torque constant,
rotor inertia, and viscous friction vary across broad log-uniform ranges. From
`K` scored support episodes on a newly encountered plant, `finetune` adapts the
controller's weights using meta-learned initialization and per-parameter
`learned_sgd` step sizes:

    ctrl    = at(filters, 'error') >> flat >> up >> stack(rnn, n=LAYERS) >> readout
    plant   = identified(motor, ORDER)
    episode = scan(observed_loop(ctrl >> plant), boundary='episode')
    task    = externalize(episode, 'pipe.motor')
    maml    = train_step(batch(finetune(train_step(task, mse, learned_sgd(lr)))), mse, adam)

`externalize(rollout, 'pipe.motor')` moves the motor's mechanical parameters out
of `param` and into the per-episode input, scoping inner adaptation to the
controller weights without optimizer masking.

`observed_loop` and `identified` form an online system-identification feedback
loop. `identified` wraps `Motor` with recursive least-squares estimation over
observed `(omega_t, v_t) -> omega_{t+1}` transitions and emits
`Struct(output=measurement, belief=theta)`. `observed_loop` subtracts
`measurement` from the velocity reference to form `error` while forwarding
`belief` alongside it. At `'episode'` boundaries, `state_reinit` resets filter
and recurrent states and `identified`'s boundary hook resets mechanical state
while retaining the accumulated RLS fit across support and query episodes.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax.transforms.learning import learned_sgd
from nodejax import (node, trained, PNode, Node, Leaf, stack, scan, batch, at, state_reinit,
                     train_step, finetune, externalize, observed_loop, KeyStream, nn)
from nodejax.struct import Struct

DT, T = 0.05, 80
HIDDEN, LAYERS = 8, 2
ORDER = 1
K, TASKS = 5, 8
META_STEPS = 1200
INNER_LR0, META_LR = 0.02, 1e-3


@node
def Motor(dt: float, resistance: float = 1.0, inductance: float = 0.2,
          ke: float = 1.0, v_max: float = 6.0) -> Node:
    """Simulates a voltage-saturated DC motor whose mechanical coefficients live in `param` for domain randomization."""

    def param(kt=1.0, inertia=0.5, friction=0.5) -> Struct:
        return Struct(kt=jnp.asarray(kt), inertia=jnp.asarray(inertia),
                      friction=jnp.asarray(friction))

    def init(node, param: Struct) -> Struct:
        z = jnp.zeros_like(node.input)
        return Struct(current=z, omega=z)

    def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        v = jnp.clip(input, -v_max, v_max)
        di = (v - resistance * state.current - ke * state.omega) / inductance
        domega = (param.kt * state.current - param.friction * state.omega) / param.inertia
        new = Struct(current=state.current + dt * di,
                     omega=state.omega + dt * domega)
        return new, new.omega

    return Leaf(apply, init=init, param=param)


@node
def Filters(dt: float) -> PNode:
    """Computes `[error, integral, step_difference]` features from scalar tracking error.

    Emitting the raw per-step difference `input - state.prev` instead of dividing
    by `dt = 0.05` keeps all three filter outputs on comparable numerical scales
    before the shared `tanh` input projection.
    """

    def init(node):
        z = jnp.zeros_like(node.input)
        return Struct(i=z, prev=z)

    def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
        i = state.i + dt * input
        d = input - state.prev
        return Struct(i=i, prev=input), jnp.stack([input, i, d])

    return Leaf(apply, init=init)


# Concatenates all array leaves of an input PyTree into one flat feature vector.
flat = Leaf(lambda input: jnp.concatenate(
    [jnp.ravel(leaf) for leaf in jax.tree.leaves(input)]), name='flat')


@node
def Up(n_in: int, hidden: int) -> Node:
    """Projects `n_in` concatenated filter and belief features to width `hidden`."""

    def param(rng: KeyStream) -> Struct:
        return Struct(win=0.5 * jax.random.normal(rng.next(), (n_in, hidden))
                      / jnp.sqrt(n_in))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.win

    return Leaf(apply, param=param)


@node
def Readout(hidden: int) -> Node:
    """Maps a `hidden`-dimensional vector to a scalar voltage command via an affine projection."""

    def param(rng: KeyStream) -> Struct:
        return Struct(w=0.1 * jax.random.normal(rng.next(), (hidden,)), b=jnp.zeros(()))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return param.w @ input + param.b

    return Leaf(apply, param=param)


@node
def identified(plant: Node, order: int) -> Node:
    """Wraps `plant` with recursive least-squares identification, emitting `Struct(output, belief)`.

    Each step advances `plant` and updates an `order`-lag linear regression fit
    `theta` with precision matrix `P` from features `[prev_outputs..., input]` to
    the new plant output. At `'episode'` boundaries, `plant` state and the output
    lag buffer reset while `rls` carries forward so accumulated plant estimates
    persist across support and query rollouts.
    """

    def init(node, param: Struct) -> Struct:
        return Struct(inner=plant.bind(param).init(input=node.input),
                      prev=jnp.zeros(order),
                      rls=Struct(theta=jnp.zeros(order + 1), P=jnp.eye(order + 1)))

    def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, Struct]:
        x = jnp.append(state.prev, input)
        new_inner, out = plant.apply(param, state.inner, input)
        Px = state.rls.P @ x
        gain = Px / (1.0 + x @ Px)
        rls = Struct(theta=state.rls.theta + gain * (out - x @ state.rls.theta),
                     P=state.rls.P - jnp.outer(gain, Px))
        return (Struct(inner=new_inner, prev=jnp.append(state.prev[1:], out), rls=rls),
                Struct(output=out, belief=rls.theta))

    def keep_the_fit(carried: Struct, init: Struct, decided: Struct) -> Struct:
        return init.replace(rls=carried.rls)

    def build_param(param: Struct) -> Struct:
        return plant.parameterize(param).param

    return Leaf(apply, param=build_param, init=init, name=plant.name,
                boundary={'episode': keep_the_fit})


def Task() -> Node:
    """Builds a single-episode closed-loop rollout with externalized motor parameters.

    `state_reinit` resets filter and RNN states at `'episode'` boundaries while
    preserving the RLS identifier fit and observer belief register across support
    and query episodes. `externalize` moves `pipe.motor` parameters into the episode
    input so meta-training adapts controller parameters alone.
    """
    pipe = (
        at(state_reinit(Filters(DT), 'episode'), 'error')
        >> flat
        >> Up(3 + ORDER + 1, HIDDEN)
        >> state_reinit(
            stack(nn.RNN(HIDDEN).with_input(jnp.zeros(HIDDEN)), n=LAYERS),
            'episode')
        >> Readout(HIDDEN)
        >> identified(Motor(DT), ORDER)
    )
    rollout = scan(observed_loop(pipe, belief_spec=jnp.zeros(ORDER + 1)),
                   boundary='episode')
    # `observed_loop` nests the chain under `pipe` alongside its feedback registers.
    return externalize(rollout, 'pipe.motor')


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean squared velocity tracking error."""
    return jnp.mean((pred - target) ** 2)


DOMAIN = {
    'kt': (0.5, 2.0),
    'inertia': (0.15, 1.5),
    'friction': (0.2, 1.2),
}


def make_tasks(
    rs: np.random.RandomState, n_tasks: int, k: int,
) -> tuple[Struct, Struct, jax.Array]:
    """Samples `n_tasks` log-uniform motor plants from `DOMAIN` with `k` support episodes and one query episode each."""

    def draw(lo, hi, shape):
        return np.exp(rs.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32)

    plant = Struct(**{nm: draw(lo, hi, (n_tasks,)) for nm, (lo, hi) in DOMAIN.items()})
    sup_refs = (rs.uniform(0.5, 1.5, size=(n_tasks, k, 1)).astype(np.float32)
                * np.ones((1, 1, T), np.float32))
    q_refs = (rs.uniform(0.5, 1.5, size=(n_tasks, 1)).astype(np.float32)
              * np.ones((1, T), np.float32))

    def per_episode(x):
        return jnp.broadcast_to(jnp.asarray(x)[:, None], (n_tasks, k))

    sup_plant = Struct(**{nm: per_episode(v) for nm, v in plant.__items__})
    q_plant = Struct(**{nm: jnp.asarray(v) for nm, v in plant.__items__})
    tasks = Struct(
        support=Struct(
            motor=sup_plant,
            input=jnp.asarray(sup_refs),
            target=jnp.asarray(sup_refs),
        ),
        query=Struct(motor=q_plant, input=jnp.asarray(q_refs)))
    return plant, tasks, jnp.asarray(q_refs)


def plot_tuning(fname: str, title: str, plant: Struct, q_t: jax.Array,
                adapted: jax.Array, unadapted: jax.Array,
                random_adapted: jax.Array) -> str:
    """Saves a 2x3 trajectory plot comparing untuned, random-init tuned, and meta-tuned responses on held-out plants."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(T) * DT
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        ref = float(q_t[i, 0])
        ax.axhspan(ref - 0.05, ref + 0.05, color='k', alpha=0.07,
                   label='reference ± 0.05')
        ax.plot(t, q_t[i], 'k:', lw=2, label='reference')
        ax.plot(t, np.asarray(random_adapted)[i], color='0.75',
                label='random init, tuned')
        ax.plot(t, np.asarray(unadapted)[i], 'C0--', lw=1,
                label='meta init, untuned (may leave frame)')
        ax.plot(t, np.asarray(adapted)[i], 'C1', label='meta init, tuned')
        ax.set_ylim(-0.25, 2.0)
        ax.set_title(f'kt={plant.kt[i]:.2f}  J={plant.inertia[i]:.2f}  '
                     f'b={plant.friction[i]:.2f}', fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('t [s]')
    for ax in axs[:, 0]:
        ax.set_ylabel('velocity')
    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'plots', fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def test_meta_controller_adapts() -> None:
    """Verifies that `K` support episodes adapt the meta-learned controller to unseen plants better than untuned or random-init baselines."""
    task = Task()
    adapt = finetune(train_step(task, mse, learned_sgd(INNER_LR0)))
    _, _shape_tasks, _ = make_tasks(np.random.RandomState(1), TASKS, k=K)
    model = batch(adapt).with_input(_shape_tasks).parameterize(rng=jax.random.PRNGKey(0))
    trainer = train_step(model, mse, optax.adam(META_LR))

    # Reshape flat task draws to add outer `(META_STEPS, TASKS)` axes for `trained` and `batch`.
    _, train_tasks, q_t = make_tasks(np.random.RandomState(0),
                                     META_STEPS * TASKS, k=K)

    def fold(x):
        return jax.tree.map(lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:]), x)
    folded = fold(train_tasks)
    final, aux = trained(trainer).apply(
        support=folded.support,
        query=folded.query,
        target=fold(q_t),
    )

    assert jnp.all(jnp.isfinite(aux.loss))
    assert jnp.mean(aux.loss[-20:]) < 0.5 * jnp.mean(aux.loss[:20])

    plant, tasks, q_t = make_tasks(np.random.RandomState(99), TASKS, k=K)

    _, adapted = final.apply(bundle=tasks)
    init_model = batch(task).with_input(tasks.query).bind(final.param.objective.model)
    _, unadapted = init_model.initialize().apply(bundle=tasks.query)
    random_adapted = batch(adapt).bind(model.param).apply(bundle=tasks)

    # Render the diagnostic plot before checking assertions so failures leave an inspectable figure.
    out = plot_tuning('meta_controller_sgd.png',
                      'meta-learned controller: five sgd support episodes '
                      'tune to an unseen plant',
                      plant, q_t, adapted, unadapted, random_adapted)
    assert os.path.exists(out)

    mse_adapted = mse(adapted, q_t)
    mse_unadapted = mse(unadapted, q_t)
    mse_random = mse(random_adapted, q_t)

    # Thresholds accommodate float32 drift between x86_64 (~0.74) and arm64 (<0.60) on identical PRNG keys.
    assert mse_adapted < 0.85 * mse_unadapted
    assert mse_adapted < 0.4 * mse_random

    # Steady-state error over the second half of the horizon isolates tracking from initial slew-rate limits.
    settled = mse(adapted[:, T // 2:], q_t[:, T // 2:])
    bias = jnp.mean(adapted[:, -20:] - q_t[:, -20:], axis=1)
    assert settled < 2e-2, settled
    assert jnp.mean(jnp.abs(bias)) < 0.05, bias

    print(f"\n[meta-controller] meta-loss {jnp.mean(aux.loss[:20]):.4f} -> "
          f"{jnp.mean(aux.loss[-20:]):.4f} over {META_STEPS} steps | held-out query mse: "
          f"adapted {mse_adapted:.4f} (settled {settled:.5f}, worst bias "
          f"{jnp.max(jnp.abs(bias)):.4f}), unadapted {mse_unadapted:.4f}, "
          f"random-init adapted {mse_random:.4f}")
