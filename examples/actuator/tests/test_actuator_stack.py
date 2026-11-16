"""Exercises the closed-loop `ActuatorStack` coupled to `Battery` sag and `Mechanical` load dynamics.

The controller normalizes PWM commands against its filtered bus-voltage estimate while the
electrical motor sees PWM scaled by true battery voltage. Before each step the environment reads
open-circuit sag voltage via `Battery.voltage`, and after each step it discharges the battery with
reported electrical power.
"""

import jax
import jax.numpy as jnp

from nodejax import Node, PNode, scan, scanned, Composite, ambient, replace_by_path
from nodejax.core.types import PyTree
from nodejax.struct import Struct
from nodejax.control import EMA, PID
from examples.actuator import (DQ, Electrical, Mechanical,
                                       Battery, DeratingThermal, FET, Noisy,
                                       ModelEstimator, foc_current_model,
                                       CurrentController, torque_command,
                                       VelocityCommand, CurrentSensor,
                                       Encoder, Observer, ActuatorStack)

DT = 1e-4


def Environment(actuator: Node, battery: Node, mechanical: Node) -> Node:
    """Couples `actuator`, `battery`, and `mechanical` into a closed-loop step returning torque."""
    members = Composite(actuator=actuator, battery=battery, mechanical=mechanical)

    def apply(self, command, load):
        out = self.actuator(mechanical=self.mechanical.state, command=command,
                            bus_voltage=self.battery.voltage())
        self.battery(out.power)
        self.mechanical(torque=out.torque, load=load)
        return out.torque

    return members(apply, name='env')


def build_env(command_ctrl = None, dt: float = DT) -> tuple[PNode, PNode]:
    """Builds a parameterized closed-loop actuator environment and its nominal motor model."""
    with ambient(dt=dt):
        motor = Electrical().parameterize(resistance=0.24, inductance_d=2e-4, inductance_q=3e-4, kt=1.2, pole_pairs=16.0, slots=36.0, cogging=0.0)

        current_filter = (
            CurrentSensor()(noise_std=0.1)
            >> EMA(warm=True)(tau=2e-4)
        )
        current_estimator = ModelEstimator(filter=current_filter, model_fn=foc_current_model()).parameterize(mix=Struct(tau=0.0))
        feedback_controller = PID().parameterize(kp=0.5, ki=200.0, integral_limit=48.0)
        power_stage = FET().parameterize(r_th=2.0, c_th=0.5, limit=80.0, r_dson=0.02)
        bus_estimator = (
            Noisy()(noise_std=0.2)
            >> EMA(warm=True)(tau=1e-3)
        )

        current_controller = (
            CurrentController(motor=motor, estimator=current_estimator, controller=feedback_controller, fets=power_stage, bus_est=bus_estimator)
            .parameterize(ff=Struct(r=0.5, bemf=0.5, l=0.0), limit=Struct(limit=50.0))
        )

        if command_ctrl is None:
            command_controller = VelocityCommand(PID().parameterize(kp=1.0, ki=10.0, integral_limit=40.0))
        else:
            command_controller = command_ctrl

        battery = Battery().parameterize(voltage_max=48.0, voltage_min=36.0, capacity=1.0e6)
        mechanical_estimator = (
            Encoder()
            >> Observer()(tau_pos=1.0, tau_vel=100.0)
        )
        motor_thermal = DeratingThermal().parameterize(r_th=0.5, c_th=50.0, limit=120.0)

        actuator = ActuatorStack(mechanical_est=mechanical_estimator, command_ctrl=command_controller, current_ctrl=current_controller, motor=motor, motor_thermal=motor_thermal).parameterize()
        mechanical = Mechanical().parameterize(inertia=0.1, friction=0.2)
        environment = Environment(actuator=actuator, battery=battery, mechanical=mechanical).parameterize()

        return environment, motor


# Warm EMA filters prime their initial state from a concrete at-rest input.
AT_REST = Struct(command=0.0, load=0.0)


def simulate(node, n: int, command: jax.Array, key: jax.Array=0):
    """Runs an `n`-step recorded rollout under `command` and returns paired outputs and states."""
    rollout = scanned(node.node, record=True).bind(node.param)
    cmds = jnp.broadcast_to(jnp.asarray(command, dtype=jnp.float32), (n,))
    out, aux = rollout.apply(
        rng=jax.random.PRNGKey(key), command=cmds,
        load=jnp.zeros_like(cmds))
    return Struct(output=out, state=aux.state)


def test_stack_assembly():
    env, _ = build_env()
    state = env.init(rng=jax.random.PRNGKey(0), input=AT_REST)

    act = state.actuator
    assert type(act.current_ctrl.pwm_prev) is DQ
    assert type(act.current_ctrl.estimator.prev) is DQ
    assert state.battery == 1.0
    assert act.current_ctrl.fets == 25.0
    assert act.motor_thermal == 25.0
    assert act.mechanical_est.observer.velocity == 0.0
    # Each stochastic sensor member receives an independent PRNG stream split from the root key.
    keys = [act.current_ctrl.bus_est.noisy.rng, act.mechanical_est.encoder.rng,
            act.current_ctrl.estimator.filter.current_sensor.rng]
    assert jnp.any(keys[0] != keys[1]) and jnp.any(keys[1] != keys[2])

    assert len(jax.tree.leaves(state)) >= 20

    paths = {jax.tree_util.keystr(p)
             for p, _ in jax.tree_util.tree_flatten_with_path(env)[0]}
    assert '.actuator.current_ctrl.controller.kp' in paths
    assert '.battery.capacity' in paths


def test_torque_command():
    """Verifies closed-loop torque tracking under `torque_command` control."""
    env, _ = build_env(command_ctrl=torque_command())
    traj = simulate(env, 1500, command=3.0)

    # Smooth thermal-derating sigmoids attenuate torque by roughly 10% at ambient temperature,
    # settling delivered torque near 2.7 Nm for a 3.0 Nm command.
    assert jnp.abs(jnp.mean(traj.output[-500:]) - 3.0) < 0.5
    assert traj.state.mechanical.velocity[-1] > 1.0


def test_velocity_command():
    env, _ = build_env()
    traj = simulate(env, 4000, command=3.0)
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 3.0) < 0.5


def test_pwm_is_normalized():
    """Verifies that the current controller normalizes PWM output within the unit `DQ` disk."""
    env, _ = build_env()
    traj = simulate(env, 2000, command=50.0)

    pwm = traj.state.actuator.current_ctrl.pwm_prev
    assert jnp.all(pwm.norm() <= 1.0 + 1e-4)
    assert jnp.max(pwm.norm()) > 0.9


def test_battery_sags_and_the_stack_feels_it():
    """Verifies that finite battery capacity drains under load, lowers bus voltage, and updates the bus-voltage estimator."""
    env, _ = build_env()
    env = replace_by_path(env, {'.battery.capacity': 10.0})
    traj = simulate(env, 4000, command=3.0)

    charge = traj.state.battery
    assert charge[-1] < 0.95
    # Momentary regenerative braking can briefly recharge the battery during transients.
    est_v = traj.state.actuator.current_ctrl.bus_est.ema
    assert est_v[-1] < 47.0
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 3.0) < 0.5


def test_model_blend_runs():
    """Verifies stability of `ModelEstimator` when `tau > 0` blends the `foc_current_model` prediction.

    Recursion through the previous current estimate carries loop gain proportional to
    `L / (R * dt)` (~8.3 here), so model weights above ~0.12 diverge while `0.05 * DT` remains stable.
    """
    env, _ = build_env()
    env = replace_by_path(env, {'.actuator.current_ctrl.estimator.mix.tau': 0.05 * DT})
    traj = simulate(env, 2000, command=2.0)

    assert jnp.all(jnp.isfinite(traj.output))
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 2.0) < 0.7


def test_thermals_heat_and_derate():
    """Verifies that ohmic dissipation heats both thermal models and that a tight `FET` limit derates current."""
    env, _ = build_env(command_ctrl=torque_command())
    traj = simulate(env, 3000, command=30.0)
    # High winding thermal mass warms slowly while the low-mass FET stage heats rapidly and derates current.
    assert traj.state.actuator.motor_thermal[-1] > 25.5
    assert traj.state.actuator.current_ctrl.fets[-1] > 29.0

    baseline, _ = build_env(command_ctrl=torque_command())
    cool = replace_by_path(baseline, {'.actuator.current_ctrl.fets.limit': 200.0})
    hot = replace_by_path(baseline, {'.actuator.current_ctrl.fets.limit': 35.0})
    iq_cool = simulate(cool, 2000, command=30.0).state.actuator.motor.q
    iq_hot = simulate(hot, 2000, command=30.0).state.actuator.motor.q
    assert jnp.mean(jnp.abs(iq_hot[-500:])) < 0.8 * jnp.mean(jnp.abs(iq_cool[-500:]))


def test_streams_are_keyed():
    env, _ = build_env()
    a = simulate(env, 500, command=2.0, key=0)
    b = simulate(env, 500, command=2.0, key=0)
    c = simulate(env, 500, command=2.0, key=1)

    assert jnp.allclose(a.output, b.output)
    assert not jnp.allclose(a.output, c.output)


def punchy_command():
    """Builds a high-gain velocity `PID` controller with `integral_limit` sized against cascade windup.

    Back-calculation inside `PID` observes its own output clamp rather than downstream current or
    bus-voltage saturation, so capping steady demand at 15 Nm (just above the 11 Nm friction torque
    at 55 rad/s) prevents overshoot at the voltage rail.
    """
    return VelocityCommand(
        PID(DT).parameterize(kp=8.0, ki=40.0, integral_limit=15.0))


def step_profile(n: int=8000):
    """Generates a +/-55 rad/s velocity step profile that exercises both the 50 A current limit and the 48 V bus rail."""
    t = jnp.arange(n) * DT
    return t, jnp.where(t < 0.25, 55.0, jnp.where(t < 0.55, -55.0, 0.0))


def render_stack_panels(t: float, cmd, traj, filename: str, title: str):
    """Renders a four-panel diagnostic plot of velocity, `DQ` currents, PWM utilization, and temperatures.

    Trajectories with a trailing ensemble axis overlay one curve per member with a single legend
    entry per signal family.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def fam(ax, y, label, **kw):
        lines = ax.plot(t, y, **kw)
        lines[0].set_label(label)

    act = traj.state.actuator
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    ax.plot(t, cmd, 'k--', lw=1, label='command')
    fam(ax, traj.state.mechanical.velocity, 'true', color='C0', lw=1.2)
    fam(ax, act.mechanical_est.observer.velocity,
        'observer est (encoder-quantized)', color='C1', lw=0.7, alpha=0.6)
    ax.set_ylabel('velocity [rad/s]')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    fam(ax, act.motor.q, 'iq true', color='C0', lw=1.0)
    fam(ax, act.current_ctrl.estimator.prev.q, 'iq estimated',
        color='C1', lw=0.7, alpha=0.6)
    fam(ax, act.motor.d, 'id true', color='C2', lw=0.7, alpha=0.6)
    for lim in (50.0, -50.0):
        ax.axhline(lim, color='r', ls=':', lw=1)
    ax.set_ylabel('current [A]')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[2]
    fam(ax, act.current_ctrl.pwm_prev.norm(), '|pwm| (dq norm)', color='C0', lw=1.0)
    ax.axhline(1.0, color='r', ls=':', lw=1, label='power stage limit')
    ax.set_ylabel('pwm utilization')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[3]
    fam(ax, act.current_ctrl.fets, 'fet temperature', color='C0', lw=1.0)
    fam(ax, act.motor_thermal, 'winding temperature', color='C1', lw=1.0)
    ax.set_ylabel('temperature [C]')
    ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def test_step_response_visual():
    """Verifies that the high-gain velocity step engages both current and PWM limits without windup and writes `step_response.png`."""
    import os

    env, _ = build_env(command_ctrl=punchy_command())
    t, cmd = step_profile()
    traj = simulate(env, len(t), command=cmd)

    act = traj.state.actuator
    iq = act.motor.q
    pwm = act.current_ctrl.pwm_prev.norm()

    # Acceleration engages the 50 A current limit, cruise sustains the unit PWM rail, and anti-windup holds 55 rad/s.
    assert jnp.max(jnp.abs(iq)) > 45.0
    assert jnp.mean(pwm > 0.95) > 0.1
    assert jnp.abs(jnp.mean(traj.state.mechanical.velocity[1800:2500]) - 55.0) < 1.0

    out = render_stack_panels(
        t, cmd, traj, 'step_response.png',
        'actuator stack: aggressive step response against both limits')
    assert os.path.exists(out)


def census(title: str, tree: PyTree):
    """Prints every leaf of `tree` with its keyed path, dtype, and shape, returning the flattened `(path, leaf)` pairs."""
    keyed = jax.tree_util.tree_flatten_with_path(tree)[0]
    print(f"\n=== {title}: {len(keyed)} leaves, "
          f"{sum(jnp.size(l) for _, l in keyed)} scalars ===")
    for path, leaf in keyed:
        spec = f"{jnp.asarray(leaf).dtype}{list(jnp.shape(leaf))}"
        print(f"  {jax.tree_util.keystr(path):72s} {spec}")
    return keyed


def test_structure_census(capsys=None):
    """Verifies that flattened PyTree key paths on `env` resolve as attribute chains on `env.param`."""
    env, _ = build_env()
    state = env.init(rng=jax.random.PRNGKey(0), input=AT_REST)

    params = census('params (the object graph)', env)
    states = census('state (one tick of the world)', state)

    # Keyed PyTree paths resolve directly via attribute lookup on env.param.
    import functools
    for key_path, leaf in params[:6]:
        chain = jax.tree_util.keystr(key_path).split('.')[1:]
        assert functools.reduce(getattr, chain, env.param) is leaf
    assert len(params) > 25 and len(states) > 15


def test_dt_is_one_ambient_knob():
    """Verifies that halving ambient `dt` while doubling step count reproduces the same physical velocity trajectory."""
    fine, _ = build_env(dt=DT / 2)
    coarse, _ = build_env()

    v_fine = simulate(fine, 4000, command=2.0).state.mechanical.velocity[-1]
    v_coarse = simulate(coarse, 2000, command=2.0).state.mechanical.velocity[-1]
    assert v_coarse > 1.5 and jnp.abs(v_fine - v_coarse) < 0.4


def test_model_mismatch_ensemble_visual():
    """Evaluates a 10-member `ensemble` where each controller's internal motor model is perturbed by +/-10% against a nominal plant.

    `ensemble` maps perturbed parameters and splits sensor-noise keys across members, isolating
    plant-model mismatch around the nominal member 0 baseline and saving `model_mismatch_ensemble.png`.
    """
    import os
    from nodejax import ensemble

    N = 10
    env, _ = build_env(command_ctrl=punchy_command())

    factors = jnp.concatenate([jnp.ones((1, 4)),
                               jax.random.uniform(jax.random.PRNGKey(7), (N - 1, 4),
                                                  minval=0.9, maxval=1.1)])
    census('params (the tree the corruption edits)', env)
    census('state (one member, before stacking)', env.init(rng=jax.random.PRNGKey(0), input=AT_REST))

    def corrupt(f):
        MODEL = '.actuator.current_ctrl.motor'
        return replace_by_path(env.param, {
            f'{MODEL}.resistance': lambda v: v * f[0],
            f'{MODEL}.kt': lambda v: v * f[1],
            f'{MODEL}.inductance_d': lambda v: v * f[2],
            f'{MODEL}.inductance_q': lambda v: v * f[3]})

    rollout = scanned(ensemble(env.node, n=len(factors)), record=True).bind(
        jax.vmap(corrupt)(factors))

    t, cmd = step_profile()
    out, aux = rollout.apply(
        rng=jax.random.PRNGKey(0), command=cmd,
        load=jnp.zeros_like(cmd))
    traj = Struct(output=out, state=aux.state)

    vel = traj.state.mechanical.velocity
    assert vel.shape == (len(t), N)
    assert jnp.all(jnp.isfinite(vel))
    # Feedback keeps all perturbed controllers within 3 rad/s of the 55 rad/s cruise target while leaving model spread measurable.
    assert jnp.abs(jnp.mean(vel[1800:2500, 0]) - 55.0) < 1.0
    assert jnp.all(jnp.abs(jnp.mean(vel[1800:2500], axis=0) - 55.0) < 3.0)
    assert jnp.std(vel[2500]) > 0.0

    out = render_stack_panels(
        t, cmd, traj, 'model_mismatch_ensemble.png',
        'the same step, ten controllers whose R/L/Kt model is off by up to 10%')
    assert os.path.exists(out)
