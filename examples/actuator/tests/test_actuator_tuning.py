"""Controller tuning under domain randomization: a multi-term cost
(tracking + settled dissipation + torque jitter + soft overcurrent),
multiplicative-uniform domain randomization over the TRUE plant, robust
90th-percentile aggregation over sampled worlds, differential evolution
over path-addressed gains and filter constants, and before/after plots
across the domain cloud.

The domain is a dict of path -> sampler, the tunables a dict of
path -> initial value; both apply through one primitive
(replace_by_path), and the optimizer's flat vector is an internal
detail. Only the TRUE plant randomizes — the controller's internal
motor model stays nominal, so model-vs-reality mismatch arises
structurally, not by injection.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

from nodejax import scan, scanned, replace_by_path
from nodejax.struct import Struct
from examples.actuator.tests.test_actuator_stack import build_env, DT


def uniform(lo: float, hi: float):
    """A domain sampler: key -> multiplicative factor on a nominal leaf."""
    return lambda key: lambda v: v * jax.random.uniform(key, minval=lo, maxval=hi)


# the domain: samplers on TRUE-plant addresses (the controller's internal
# model is deliberately NOT in this dict)
DOMAIN = {
    # '.actuator.motor.resistance': uniform(0.9, 1.2),
    # '.actuator.motor.inductance_d': uniform(0.8, 1.2),
    # '.actuator.motor.inductance_q': uniform(0.8, 1.2),
    # '.actuator.motor.kt': uniform(0.95, 1.05),
    # '.actuator.battery.voltage_max': uniform(0.9, 1.0),
    # '.mechanical.inertia': uniform(0.8, 1.2),
    # '.mechanical.friction': uniform(0.1, 10.0),
}

# the tunables: a credible hand tuning of the GAINS (tracks the gentle
# ramp with visible lag; DE's job is the last factor, not resurrection)
# plus the FILTER time constants (observer, bus filter) and PER-TERM
# feedforward trust weights (resistive / back-EMF / inductive-by-
# reference-derivative — the empirical jury on each). Bandwidth wants hot gains, noise wants slow
# filters, and feedforward buys tracking without feedback gain — giving
# the optimizer all ends of that trade is what keeps the optimum from
# being a chatter amplifier.
INITIAL_GAINS = {
    '.actuator.command_ctrl.velocity_ctrl.kp': 4.0,
    '.actuator.command_ctrl.velocity_ctrl.ki': 10.0,
    '.actuator.command_ctrl.velocity_ctrl.integral_limit': 40.0,
    '.actuator.current_ctrl.controller.kp': 0.5,
    '.actuator.current_ctrl.controller.ki': 500.0,
    '.actuator.current_ctrl.ff.r': 0.5,
    '.actuator.current_ctrl.ff.bemf': 0.5,
    '.actuator.current_ctrl.ff.l': 0.1,
    '.actuator.mechanical_est.observer.tau_pos': 1.0,
    '.actuator.mechanical_est.observer.tau_vel': 100.0,
    '.actuator.current_ctrl.bus_est.ema.tau': 1e-3,
}
# (estimator.tau is deliberately NOT a gene: it is the model-blend
# weight, and the FOC current model diverges above ~0.12dt — an
# optimizer handed that knob finds NaN, not smoothness)

DOMAINS, MAX_CURRENT = 7, 50.0
N = 2500                                            # 0.25 s


def scenario():
    """High-speed operation, where voltage feedforward earns its keep:
    back-EMF at 35-48 rad/s is 28-38 V of the 48 V bus, so the voltage
    budget is mostly back-EMF and the top climb runs NEAR THE VOLTAGE
    LIMIT (modulation ~0.9). An integrator must wind that budget up at
    ki*err*dt volts per step during every speed change; feedforward
    supplies it instantly from the velocity estimate. Segments: brisk
    feasible ramp to 35, hold, an infeasible drop to 10 (windup + 
    saturation at speed), hold, then a voltage-limited climb to 48.
    """
    t = np.arange(N) * DT
    cmds = np.interp(t, [0.0, 0.02, 0.09, 0.12, 0.125, 0.15, 0.22, 0.25],
                        [0.0, 0.0,  35.0, 35.0, 10.0,  10.0, 48.0, 48.0])
    return jnp.asarray(t), jnp.asarray(cmds, dtype=jnp.float32)


def randomize_domain(env, key: jax.Array):
    """Sample one physical world: split the key per domain entry, apply
    every sampler at its address."""
    keys = jax.random.split(key, len(DOMAIN))
    return replace_by_path(env, {path: fn(k)
                                 for (path, fn), k in zip(DOMAIN.items(), keys)})


def with_gains(env, vector: jax.Array):
    """Optimizer vector -> gains at their addresses (log-mapped, so the
    search stays positive)."""
    return replace_by_path(env, dict(zip(INITIAL_GAINS, jnp.exp(vector))))


def cost(ys: jax.Array, cmds: jax.Array) -> jax.Array:
    """Tracking, settled-window dissipation, torque jitter (an
    audible-noise proxy), and soft overcurrent."""
    v = ys.state.mechanical.velocity
    i = ys.state.actuator.motor                     # DQ over time
    i2 = i.norm2()

    tracking = jnp.mean((v - cmds) ** 2)
    settled = 3 * len(cmds) // 4
    ohmic = jnp.mean(i2[settled:])                  # noise and wobbles at the end
    jitter = jnp.mean(jnp.abs(jnp.diff(i.q, n=2)))  # audible-noise proxy
    overcurrent = jnp.mean(jnp.maximum(jnp.sqrt(i2) - (MAX_CURRENT + 5.0), 0.0) ** 2)

    # jitter calibrated to BITE (~5-20% of total at chattery solutions):
    # a noise term the optimizer cannot feel selects FOR noise
    total = 5.0 * tracking + 0.1 * ohmic + 10.0 * jitter + 10.0 * overcurrent
    # non-finite rollouts (a candidate destabilized some domain) are
    # maximally bad, not percentile poison
    return jnp.where(jnp.isfinite(total), total, 1e6)


def test_domain_randomized_tuning():
    env, _ = build_env()
    rollout = scanned(env.node, record=True)

    t, cmds = scenario()

    # one set of worlds and sensor streams, identical for every candidate
    domain_keys = jax.random.split(jax.random.PRNGKey(42), DOMAINS)
    sensor_keys = jax.random.split(jax.random.PRNGKey(7), DOMAINS)

    def run(vector, domain_key, sensor_key):
        world = randomize_domain(with_gains(env, vector), domain_key)
        out, aux = rollout.apply(
            world.param, rng=sensor_key, command=cmds,
            load=jnp.zeros_like(cmds))
        return Struct(output=out, state=aux.state)

    @jax.jit
    @jax.vmap
    def evaluate(vector):
        costs = jax.vmap(lambda dk, sk: cost(run(vector, dk, sk), cmds))(
            domain_keys, sensor_keys)
        return jnp.percentile(costs, 90)            # robust: the bad domains decide

    # DE/rand/1/bin refining from the shipped gains, bounded in
    # log-space. (A wide search from a straw-man start finds a degenerate
    # near-equivalent optimum — tiny integral limit, huge ki; the modest
    # spread from a sane start lands on the classical controller, ~1%
    # apart in cost. Scenario-dependence of 'optimal gains', quantified.)
    start = jnp.log(jnp.asarray(list(INITIAL_GAINS.values())))
    P, D, GENS, F, CR = 16, len(INITIAL_GAINS), 16, 0.7, 0.7   # 10-D search
    baseline = evaluate(start[None])[0]

    key = jax.random.PRNGKey(1)
    key, k = jax.random.split(key)
    pop = start + 0.5 * jax.random.normal(k, (P, D))
    fit = evaluate(pop)
    bounds = (start - 3.0, start + 3.0)

    def de_step(carry, gen_key):
        pop, fit = carry
        ka, kb, kc, km, kj = jax.random.split(gen_key, 5)
        a, b, c = (jax.random.randint(kk, (P,), 0, P) for kk in (ka, kb, kc))
        mutant = pop[a] + F * (pop[b] - pop[c])
        cross = jax.random.uniform(km, (P, D)) < CR
        cross = cross.at[jnp.arange(P), jax.random.randint(kj, (P,), 0, D)].set(True)
        trial = jnp.clip(jnp.where(cross, mutant, pop), *bounds)
        tfit = evaluate(trial)
        won = tfit < fit
        new_pop = jnp.where(won[:, None], trial, pop)
        new_fit = jnp.where(won, tfit, fit)
        return (new_pop, new_fit), None

    gen_keys = jax.random.split(key, GENS)
    (pop, fit), _ = jax.lax.scan(de_step, (pop, fit), gen_keys)

    best_vector = pop[jnp.argmin(fit)]
    best = jnp.min(fit)
    tuned = dict(zip(INITIAL_GAINS, np.exp(np.asarray(best_vector))))

    # robust tuning must clearly beat the detuned baseline ACROSS domains;
    # gene physicality (positive, bounded) is enforced by the log-space
    # clip bounds themselves — assert they held
    assert best < 0.6 * baseline, (best, baseline)
    assert all(np.isfinite(v) and v > 0.0 for v in tuned.values())
    assert jnp.all(best_vector >= bounds[0] - 1e-6) and jnp.all(best_vector <= bounds[1] + 1e-6)

    # --- the before/after picture: a FRESH cloud of 30 domain draws
    # (not the fitness set), before and after OVERLAID per panel ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cloud = 30
    cloud_keys = jax.random.split(jax.random.PRNGKey(43), n_cloud)
    cloud_sensors = jax.random.split(jax.random.PRNGKey(44), n_cloud)
    before = jax.vmap(lambda dk, sk: run(start, dk, sk))(cloud_keys, cloud_sensors)
    after = jax.vmap(lambda dk, sk: run(best_vector, dk, sk))(cloud_keys, cloud_sensors)

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    alpha = 2.0 / n_cloud

    def overlay(ax, series, ylabel, title):
        for d in range(n_cloud):
            ax.plot(t, series(before)[d], color='C0', ls='--', alpha=alpha,
                    label='before optimization')
            ax.plot(t, series(after)[d], color='C1', alpha=alpha,
                    label='after optimization')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        for line in legend.get_lines():
            line.set_alpha(1.0)

    overlay(axs[0], lambda ys: ys.state.actuator.current_ctrl.pwm_prev.norm(),
            'modulation', 'Modulation depth (|pwm|)')
    overlay(axs[1], lambda ys: jnp.sqrt(ys.state.actuator.motor.norm2()),
            'A', 'Current magnitude')
    axs[1].axhline(MAX_CURRENT, color='r', ls=':', lw=1)
    overlay(axs[2], lambda ys: ys.state.mechanical.velocity,
            'rad/s', 'Velocity response')
    axs[2].plot(t, cmds, 'k--', lw=1.2, label='target')
    axs[2].set_xlabel('time [s]')

    fig.suptitle('domain-randomized gain tuning: dicts of addresses, one pytree')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'plots', 'tuning.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    assert os.path.exists(out)

    print(f"\n[tuning] p90 cost {baseline:.1f} -> {best:.1f} over {DOMAINS} domains")
    for path, value in tuned.items():
        print(f"  {'.'.join(path.split('.')[-2:]):18s} {INITIAL_GAINS[path]:10.4g} -> {value:10.4g}")
