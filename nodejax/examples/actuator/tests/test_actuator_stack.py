"""The full actuator stack, closed-loop: battery -> voltage estimation ->
command controller -> current controller (model-based estimation on the
modulated voltage, per-term feedforward) -> electrical motor, with
mechanics integrated at the environment level.

The est/true voltage asymmetry is modeled: the controller normalizes by
its ESTIMATED bus voltage (a noisy >> ema sensor pipeline), the motor is
driven by pwm x TRUE battery voltage, and the battery both sags (charge
state) and is read two ways (voltage METHOD before the step, discharge
apply after).
"""

import jax
import jax.numpy as jnp

from nodejax import scan, composite, ambient
from nodejax.struct import Struct
from nodejax.examples.actuator import (DQ, motor_params, motor_model_def, emotor_def, mechanical_def,
                                    battery_def, derating_thermal_def, fet_def, ema_def, noisy_def,
                                    model_estimator_def, foc_current_model,
                                    current_controller_def, torque_command_def,
                                    velocity_command_def, current_sensor_def,
                                    encoder_def, observer_def, pid_def, actuator_stack_def)

DT = 1e-4


def env_def(actuator, mechanical):
    """Environment-level closure: the actuator maps (mechanical, command)
    -> torque; the environment integrates mechanics."""
    members = dict(actuator=actuator, mechanical=mechanical)

    def apply(self, command):
        torque = self.actuator(mechanical=self.state.mechanical, command=command)
        self.mechanical(torque=torque, load=0.0)
        return torque

    return composite(apply, members=members, name='env', apply_input_spec=Struct(command=0.0))


def build_env(command_ctrl=None, capacity=jnp.inf, model_tau=0.0,
              fet_limit=80.0, dt=DT):
    """Assemble the full chain — the member tree spelled once, at the
    factories: blocks arrive as defs or constructed (bound nodes, their
    params the stored construction values). dt is AMBIENT: declared
    eligible at each factory's definition site (@ambient), supplied
    once here, filled only where a call leaves it unbound — no
    threading, no registries, explicit always wins."""
    m = motor_params(cogging=0.0)   # clean tracking; the bench tests own cogging fidelity
    with ambient(dt=dt):
        cc = current_controller_def(
            motor=motor_model_def().bind(m),   # internal model: params, no state
            estimator=model_estimator_def(
                filter=current_sensor_def(noise_std=0.1),
                model_fn=foc_current_model()).parameterize(mix=Struct(tau=model_tau)),
            controller=pid_def().parameterize(kp=0.5, ki=200.0, integral_limit=48.0),
            fets=fet_def().parameterize(r_th=2.0, c_th=0.5, limit=fet_limit),
            bus_sensor=noisy_def(0.2) >> ema_def()(tau=1e-3),
        ).parameterize(ff=Struct(r=0.5, bemf=0.5, l=0.0), limit=Struct(limit=50.0))
        command = command_ctrl if command_ctrl is not None else \
            velocity_command_def(pid_def().parameterize(kp=1.0, ki=10.0,
                                                        integral_limit=40.0))
        actuator = actuator_stack_def(
            battery=battery_def().parameterize(voltage_max=48.0, capacity=capacity),
            mechanical_est=encoder_def() >> observer_def()(tau_pos=1.0, tau_vel=100.0),
            command_ctrl=command, current_ctrl=cc,
            motor=emotor_def().bind(m),
            motor_thermal=derating_thermal_def().parameterize(
                r_th=0.5, c_th=50.0, limit=120.0)).parameterize()
        return env_def(actuator=actuator,
                       mechanical=mechanical_def().parameterize(
                           inertia=0.1, friction=0.2)).parameterize(), m


def simulate(node, n, command, key=0):
    rollout = scan(node.ndef, record=True).bind(node.param)
    cmds = jnp.broadcast_to(jnp.asarray(command, dtype=jnp.float32), (n,))
    # commands are the raw input sequence; the key rides the reserved field
    return rollout.apply(Struct(rng=jax.random.PRNGKey(key), command=cmds))


def test_stack_assembly():
    env, _ = build_env()
    state = env.init(rng=jax.random.PRNGKey(0))

    act = state.actuator
    assert isinstance(act.current_ctrl.pwm_prev, DQ)            # previous pwm
    assert isinstance(act.current_ctrl.estimator.prev, DQ)  # model blend memory
    assert act.battery == 1.0                                 # full charge
    assert act.current_ctrl.fets == 25.0                      # fets at ambient
    assert act.motor_thermal == 25.0
    assert act.mechanical_est.observer.velocity == 0.0
    # one sensor stream per stochastic member, all split from one key
    keys = [act.current_ctrl.bus_sensor.noisy.rng, act.mechanical_est.encoder.rng,
            act.current_ctrl.estimator.filter.rng]
    assert jnp.any(keys[0] != keys[1]) and jnp.any(keys[1] != keys[2])

    # the maximal-stack claim, quantified: connected stateful leaves
    assert len(jax.tree.leaves(state)) >= 20

    paths = {jax.tree_util.keystr(p)
             for p, _ in jax.tree_util.tree_flatten_with_path(env)[0]}
    assert '.actuator.current_ctrl.controller.kp' in paths
    assert '.actuator.battery.capacity' in paths


def test_torque_command():
    """Torque command -> current target -> pwm -> electrical torque."""
    env, _ = build_env(command_ctrl=torque_command_def())
    traj = simulate(env, 1500, command=3.0)

    # the sigmoid rollbacks derate ~10% even at ambient (their tails never
    # reach 1), so delivered torque sits near 2.7, not 3.0 — physics, not
    # tracking error; the tolerance accounts for it
    assert jnp.abs(jnp.mean(traj.output[-500:]) - 3.0) < 0.5
    assert traj.state.mechanical.velocity[-1] > 1.0           # spinning up


def test_velocity_command():
    env, _ = build_env()
    traj = simulate(env, 4000, command=3.0)
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 3.0) < 0.5


def test_pwm_is_normalized():
    """The actuation variable is a PWM with dq norm <= 1 — the power stage
    limit, not an absolute voltage clamp."""
    env, _ = build_env()
    traj = simulate(env, 2000, command=50.0)                  # absurd command

    pwm = traj.state.actuator.current_ctrl.pwm_prev
    assert jnp.all(pwm.norm() <= 1.0 + 1e-4)
    assert jnp.max(pwm.norm()) > 0.9                          # limit exercised


def test_battery_sags_and_the_stack_feels_it():
    """Finite capacity: charge drains with drawn power, the sag curve
    lowers the true bus voltage, and the voltage estimator tracks it."""
    env, _ = build_env(capacity=10.0)
    traj = simulate(env, 4000, command=3.0)

    charge = traj.state.actuator.battery
    assert charge[-1] < 0.95                                  # visibly drained
    # (not asserted monotone: momentary negative power is regeneration)
    est_v = traj.state.actuator.current_ctrl.bus_sensor.ema
    assert est_v[-1] < 47.0                                   # estimator follows the sag
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 3.0) < 0.5  # still tracks


def test_model_blend_runs():
    """tau > 0 engages the FOC current model (prediction from the
    previously commanded modulated voltage) in the estimator blend.

    The blend's recursion through the previous estimate has loop gain
    (model weight) * L/(R*dt) ~ 8.3 at these motor constants, so model
    weights above ~0.12 DIVERGE. Tested at a provably stable weight."""
    env, _ = build_env(model_tau=0.05 * DT)
    traj = simulate(env, 2000, command=2.0)

    assert jnp.all(jnp.isfinite(traj.output))
    assert jnp.abs(traj.state.mechanical.velocity[-1] - 2.0) < 0.7


def test_thermals_heat_and_derate():
    """Dissipation heats both thermal nodes; a tight fet temperature limit
    visibly derates the delivered current."""
    env, _ = build_env(command_ctrl=torque_command_def())
    traj = simulate(env, 3000, command=30.0)
    # windings warm slowly (heavy mass) — and the fets derate the current
    # before a full degree accumulates; the fets themselves run away fast
    assert traj.state.actuator.motor_thermal[-1] > 25.5
    assert traj.state.actuator.current_ctrl.fets[-1] > 29.0

    cool, _ = build_env(command_ctrl=torque_command_def(), fet_limit=200.0)
    hot, _ = build_env(command_ctrl=torque_command_def(), fet_limit=35.0)
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


def test_step_response_visual():
    """Aggressive step up/down velocity command, sized to exhaust the
    bus: back-EMF is ~0.8*omega volts against a 48V battery, so +/-55
    rad/s puts the CRUISE at the pwm norm-1 rail (velocity asymptotes as
    headroom vanishes — no field weakening), while each acceleration
    slams the +/-50A current limit. Renders commands vs truth vs
    estimates, both limits, thermals — tests/plots/step_response.png."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # integral_limit is the anti-windup story here: the PID's
    # back-calculation only sees its OWN clamp, not the downstream
    # current/voltage saturation (cascade windup), so the cap must sit
    # just above the legitimate steady demand — friction torque at speed
    # (0.2 * 55 = 11 Nm). At 60 the integral winds ~50 Nm of phantom
    # demand and velocity parks 3 rad/s above command at the bus rail.
    punchy = velocity_command_def(
        pid_def(DT).parameterize(kp=8.0, ki=40.0, integral_limit=15.0))
    env, _ = build_env(command_ctrl=punchy)

    n = 8000
    t = jnp.arange(n) * DT
    cmd = jnp.where(t < 0.25, 55.0, jnp.where(t < 0.55, -55.0, 0.0))
    traj = simulate(env, n, command=cmd)

    act = traj.state.actuator
    iq, i_est = act.motor.q, act.current_ctrl.estimator.prev.q
    pwm = act.current_ctrl.pwm_prev.norm()

    # the plot must actually close up on both limits — and the voltage
    # rail must be a sustained operating point, not an edge transient
    assert jnp.max(jnp.abs(iq)) > 45.0                 # current limit engaged
    assert jnp.mean(pwm > 0.95) > 0.1                  # rail held ~10%+ of the run
    # windup tamed: cruise tracks the command through the saturated ramps
    assert jnp.abs(jnp.mean(traj.state.mechanical.velocity[1800:2500]) - 55.0) < 1.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    ax.plot(t, cmd, 'k--', lw=1, label='command')
    ax.plot(t, traj.state.mechanical.velocity, lw=1.5, label='true')
    ax.plot(t, act.mechanical_est.observer.velocity, lw=0.8, alpha=0.7,
            label='observer est (encoder-quantized)')
    ax.set_ylabel('velocity [rad/s]')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    ax.plot(t, iq, lw=1.0, label='iq true')
    ax.plot(t, i_est, lw=0.8, alpha=0.7, label='iq estimated')
    for lim in (50.0, -50.0):
        ax.axhline(lim, color='r', ls=':', lw=1)
    ax.set_ylabel('current [A]')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[2]
    ax.plot(t, pwm, lw=1.0, label='|pwm| (dq norm)')
    ax.axhline(1.0, color='r', ls=':', lw=1, label='power stage limit')
    ax.set_ylabel('pwm utilization')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[3]
    ax.plot(t, act.current_ctrl.fets, lw=1.0, label='fet temperature')
    ax.plot(t, act.motor_thermal, lw=1.0, label='winding temperature')
    ax.set_ylabel('temperature [C]')
    ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('actuator stack: aggressive step response against both limits')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'plots', 'step_response.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    assert os.path.exists(out)


def test_structure_census(capsys=None):
    """Print the full param and state trees of the maximal stack — every
    leaf by keyed path, dtype and shape (run with -s to see it). Param
    addresses read as the object graph and resolve as attribute chains;
    state mirrors it member-for-member."""
    env, _ = build_env()
    state = env.init(rng=jax.random.PRNGKey(0))

    def census(title, tree):
        keyed = jax.tree_util.tree_flatten_with_path(tree)[0]
        print(f"\n=== {title}: {len(keyed)} leaves, "
              f"{sum(jnp.size(l) for _, l in keyed)} scalars ===")
        for path, leaf in keyed:
            spec = f"{jnp.asarray(leaf).dtype}{list(jnp.shape(leaf))}"
            print(f"  {jax.tree_util.keystr(path):72s} {spec}")
        return keyed

    params = census('params (the object graph)', env)
    states = census('state (one tick of the world)', state)

    # addresses ARE attribute chains: any path resolves by plain getattr
    import functools
    for key_path, leaf in params[:6]:
        chain = jax.tree_util.keystr(key_path).split('.')[1:]
        assert functools.reduce(getattr, chain, env) is leaf
    # state mirrors the stateful members only; both trees are non-trivial
    assert len(params) > 25 and len(states) > 15


def test_dt_is_one_ambient_knob():
    """dt is one ambient value at one point of use, so rebinding it
    re-times every block coherently: the same 0.2s velocity command at
    half the step size tracks the same physics (dt-in-seconds discipline
    enforced across 11 defs at once — a per-step unit bug anywhere
    splits these runs)."""
    fine, _ = build_env(dt=DT / 2)
    coarse, _ = build_env()

    v_fine = simulate(fine, 4000, command=2.0).state.mechanical.velocity[-1]
    v_coarse = simulate(coarse, 2000, command=2.0).state.mechanical.velocity[-1]
    assert v_coarse > 1.5 and jnp.abs(v_fine - v_coarse) < 0.4
