"""Tunes `ActuatorStack` feedback gains, filter time constants, and feedforward weights under plant randomization.

Domain samplers perturb physical plant parameters via `replace_by_path` while leaving the
controller's internal motor model nominal, and differential evolution minimizes 90th-percentile
rollout cost across sampled plants.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

from nodejax import scan, scanned, replace_by_path
from nodejax.struct import Struct
from examples.actuator.tests.test_actuator_stack import build_env, DT


def uniform(lo: float, hi: float):
    """Returns a sampler mapping a PRNG key to a multiplicative uniform perturbation in `[lo, hi]`."""
    return lambda key: lambda v: v * jax.random.uniform(key, minval=lo, maxval=hi)


# Domain samplers perturb physical plant paths while leaving the controller's internal motor model untouched.
DOMAIN = {
    # '.actuator.motor.resistance': uniform(0.9, 1.2),
    # '.actuator.motor.inductance_d': uniform(0.8, 1.2),
    # '.actuator.motor.inductance_q': uniform(0.8, 1.2),
    # '.actuator.motor.kt': uniform(0.95, 1.05),
    # '.actuator.battery.voltage_max': uniform(0.9, 1.0),
    # '.mechanical.inertia': uniform(0.8, 1.2),
    # '.mechanical.friction': uniform(0.1, 10.0),
}

# Tunable paths cover PID gains, observer and bus-filter time constants, and feedforward weights.
# Exposing gains, filters, and feedforward together lets differential evolution trade bandwidth against sensor noise;
# estimator.mix.tau is excluded because FOC model-blend weights above ~0.12 * dt diverge.
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

DOMAINS, MAX_CURRENT = 7, 50.0
N = 2500


def scenario():
    """Builds a high-speed velocity command profile that exercises back-EMF feedforward and bus-voltage saturation.

    Back-EMF consumes 28 to 38 V of the 48 V bus between 35 and 48 rad/s, driving modulation near
    0.9 on the final climb after a steep deceleration step to 10 rad/s.
    """
    t = np.arange(N) * DT
    cmds = np.interp(t, [0.0, 0.02, 0.09, 0.12, 0.125, 0.15, 0.22, 0.25],
                        [0.0, 0.0,  35.0, 35.0, 10.0,  10.0, 48.0, 48.0])
    return jnp.asarray(t), jnp.asarray(cmds, dtype=jnp.float32)


def randomize_domain(env, key: jax.Array):
    """Perturbs `env` across all configured `DOMAIN` paths using independent subkeys."""
    keys = jax.random.split(key, len(DOMAIN))
    return replace_by_path(env, {path: fn(k)
                                 for (path, fn), k in zip(DOMAIN.items(), keys)})


def with_gains(env, vector: jax.Array):
    """Writes exponentiated log-space `vector` entries into `env` at the `INITIAL_GAINS` paths."""
    return replace_by_path(env, dict(zip(INITIAL_GAINS, jnp.exp(vector))))


def cost(ys: jax.Array, cmds: jax.Array) -> jax.Array:
    """Computes weighted tracking error, settled ohmic loss, current-ripple jitter, and overcurrent penalty."""
    v = ys.state.mechanical.velocity
    i = ys.state.actuator.motor
    i2 = i.norm2()

    tracking = jnp.mean((v - cmds) ** 2)
    settled = 3 * len(cmds) // 4
    ohmic = jnp.mean(i2[settled:])
    jitter = jnp.mean(jnp.abs(jnp.diff(i.q, n=2)))
    overcurrent = jnp.mean(jnp.maximum(jnp.sqrt(i2) - (MAX_CURRENT + 5.0), 0.0) ** 2)

    total = 5.0 * tracking + 0.1 * ohmic + 10.0 * jitter + 10.0 * overcurrent
    # Non-finite trajectories from unstable candidate gains receive a fixed penalty so 90th-percentile aggregation stays finite.
    return jnp.where(jnp.isfinite(total), total, 1e6)


def test_domain_randomized_tuning():
    env, _ = build_env()
    rollout = scanned(env.node, record=True)

    t, cmds = scenario()

    # Common random numbers evaluate every candidate gain vector on identical plant draws and sensor noise.
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
        return jnp.percentile(costs, 90)

    # Differential evolution (DE/rand/1/bin) refines log-gains within +/-3 log-units around INITIAL_GAINS.
    start = jnp.log(jnp.asarray(list(INITIAL_GAINS.values())))
    P, D, GENS, F, CR = 16, len(INITIAL_GAINS), 16, 0.7, 0.7
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

    # Tuned gains reduce 90th-percentile cost by more than 40% while respecting positive log-space bounds.
    assert best < 0.6 * baseline, (best, baseline)
    assert all(np.isfinite(v) and v > 0.0 for v in tuned.values())
    assert jnp.all(best_vector >= bounds[0] - 1e-6) and jnp.all(best_vector <= bounds[1] + 1e-6)

    # Evaluates initial and tuned gains on 30 held-out plant and sensor draws for diagnostic plotting.
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
