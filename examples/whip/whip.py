"""Configures the whip plant at the example's physical dimensions and clocks and runs an open-loop swing.

Couples two FOC actuator stacks with quantized encoders to a two-link arm and a
PBD rope sharing a single battery pack. Under reversing shoulder strokes the
rope trails and overtakes the handle, accelerating the tip to several times the
handle speed.
"""

import math

import jax
import jax.numpy as jnp

from nodejax import Node, PNode, Struct, batch, scan, split_aux
from examples.actuator import Battery
from examples.whip.plant import (
    Arm,
    HitCost,
    IdealActuator,
    Whip,
    WhipStep,
    whip_actuator,
    whip_constraints,
    whip_particles,
    whip_rope,
)


# Actuator and rope both step once per control tick; motor currents advance
# implicitly while the rope steps coarsely with mild compliance. Above the
# control tick, the policy holds each command for N_CONTROL_TICKS.
DT = 1e-3  # control tick duration for one actuator tick and one rope step
N_ROPE_SUBSTEPS = 1  # rope steps per control tick under held torque
N_CONTROL_TICKS = 10  # control ticks per policy step under one held command
PLANT_DT = DT * N_CONTROL_TICKS
LINK_LENGTHS = (0.3, 0.3)  # upper arm and forearm lengths from shoulder to handle
N_LINKS = len(LINK_LENGTHS)
HANDLE = N_LINKS  # particle index at the end of the last link where the rope starts
SEGMENT = 0.15
N_ROPE = 8
ROTOR_INERTIA = 0.05
ROPE_MASS = 0.05
N_SOLVER_PASSES = 4
# Bending compliance increases geometrically from grip to tip; 1e4 gives roughly
# quarter stiffness at the grip and 1e6 leaves the tip nearly free.
BEND_COMPLIANCE_HANDLE = 1e4
BEND_COMPLIANCE_TIP = 1e6
# Floor height below the pivot so the arm cannot hang straight down and the rope cannot swing through.
FLOOR_HEIGHT = -0.4
ENCODER_RESOLUTION = 256.0
OBSERVER_TAU_POS = 2.0  # ticks of position filtering
OBSERVER_TAU_VEL = 5.0  # ticks of velocity filtering trading encoder quantization noise against lag
VELOCITY_GAIN = 2.5  # Nm per rad/s of velocity error before elbow chatter sets in
VELOCITY_GAIN_STIFF = 10.0  # higher gain used when exact sensing avoids quantization noise
VELOCITY_KI = 0.0  # proportional-only velocity loop
CURRENT_KP = 0.5
CURRENT_KI = 500.0
CURRENT_FILTER_TAU = 2e-3  # current sensor filter time constant in seconds
BUS_FILTER_TAU = 1e-2  # bus voltage filter time constant in seconds
BATTERY_CAPACITY = 1200.0  # joules in the shared pack supplying both joints
BUS_VOLTAGE = 48.0
TORQUE_LIMIT = 120.0  # current limit times torque constant for the ideal actuator ablation
MOTOR_KT = 1.2
TARGET = (0.0, 1.2)
PROXIMITY = 0.2
HALF_CREDIT_RADIUS = PROXIMITY * math.sqrt(math.log(2.0))  # radius where closeness reaches one half
HIT_ENERGY = 10.0  # joules (tip at 20 m/s) normalizing hit credit
HIT_POWER = 1.0  # exponent keeping hit credit linear in tip kinetic energy


def whip_at_rest(link_lengths: tuple[float, ...] = LINK_LENGTHS) -> Struct:
    """Returns the initial particle state for the whip resting straight along the x-axis."""
    return whip_particles(
        N_ROPE,
        link_lengths=link_lengths,
        segment=SEGMENT,
        rotor_inertia=ROTOR_INERTIA,
        rope_mass=ROPE_MASS,
    )


def whip_plant(
    encoder_resolution: float = ENCODER_RESOLUTION,
    tip_sensor: bool = False,
    ideal_actuator: bool = False,
    exact_encoder: bool = False,
    velocity_gain: float = VELOCITY_GAIN,
    link_lengths: tuple[float, ...] = LINK_LENGTHS,
) -> Node | PNode:
    """Builds the whip plant on an arm of `link_lengths` running at the policy clock.

    Holds each per-joint velocity command across `N_CONTROL_TICKS` actuator ticks,
    substepping the PBD rope `N_ROPE_SUBSTEPS` times per tick and summing tick costs.
    Uses batched FOC actuator stacks sharing one battery unless `ideal_actuator`
    selects the ideal torque-clamped velocity loop ablation.
    """
    actuator = (
        IdealActuator()(velocity_gain=velocity_gain, torque_limit=TORQUE_LIMIT, kt=MOTOR_KT)
        if ideal_actuator else
        whip_actuator(
            DT,
            encoder_resolution=encoder_resolution,
            observer_tau_pos=OBSERVER_TAU_POS,
            observer_tau_vel=OBSERVER_TAU_VEL,
            velocity_kp=velocity_gain,
            velocity_ki=VELOCITY_KI,
            current_kp=CURRENT_KP,
            current_ki=CURRENT_KI,
            current_filter_tau=CURRENT_FILTER_TAU,
            bus_filter_tau=BUS_FILTER_TAU,
            exact_encoder=exact_encoder,
        )
    )
    constraints = whip_constraints(
        N_ROPE,
        link_lengths=link_lengths,
        segment=SEGMENT,
        bend_compliance_handle=BEND_COMPLIANCE_HANDLE,
        bend_compliance_tip=BEND_COMPLIANCE_TIP,
    )
    rope = whip_rope(
        constraints,
        dt=DT,
        n_substeps=N_ROPE_SUBSTEPS,
        floor_height=FLOOR_HEIGHT,
        n_solver_passes=N_SOLVER_PASSES,
    )
    tick = Whip(
        batch(actuator, n=len(link_lengths)),
        Battery(DT)(voltage_max=BUS_VOLTAGE, voltage_min=0.0, capacity=BATTERY_CAPACITY),
        rope.parameterize().bind(state=whip_at_rest(link_lengths)),
        Arm(link_lengths),
        HitCost(PROXIMITY, HIT_ENERGY, HIT_POWER),
        target=TARGET,
        tip_sensor=tip_sensor,
    )
    return WhipStep(tick, N_CONTROL_TICKS)


def open_loop_swing(
    plant: Node | PNode,
    start: Struct,
    command: jax.Array,
    rng: jax.Array,
) -> Struct:
    """Scans `plant` from `start` under a `(time, joint)` velocity command sequence.

    Returns a `Struct` holding the state trajectory, applied torques, step costs,
    and true joint velocities alongside observer velocity estimates.
    """
    n_steps = command.shape[0]
    stepping = plant.parameterize()
    init_key, run_key = jax.random.split(rng)
    state = (stepping.init(rng=init_key, start=start)
             if stepping.contract.init_takes_rng else stepping.init(start=start))
    program = scan(stepping, n=n_steps).bind(state=state)
    call_kwargs = dict(command=command, disturbance=jnp.zeros((n_steps,)))
    if program.contract.apply_takes_rng:
        call_kwargs['rng'] = run_key
    _, outputs = jax.jit(program.apply)(**call_kwargs)
    outputs, _ = split_aux(outputs)
    true = stepping.mechanical(state=outputs.state)
    return Struct(
        state=outputs.state,
        action=outputs.action,
        cost=outputs.cost,
        joint_velocity=true.velocity,
        estimated_velocity=outputs.state.actuator.mechanical_est.observer.velocity,
    )


def main() -> None:
    """Runs an open-loop shoulder stroke sequence and saves the swing plot."""
    n_steps = 150
    # The shoulder strokes back and forth every half second, the elbow holds.
    sign = jnp.where((jnp.arange(n_steps) // 50) % 2 == 0, 1.0, -1.0)
    command = jnp.zeros((n_steps, N_LINKS)).at[:, 0].set(30.0 * sign)
    plant = whip_plant()
    start = plant.parameterize().start(angle=0.0, target=TARGET)
    swing = open_loop_swing(plant, start, command, jax.random.PRNGKey(1))
    tip_speed = jnp.linalg.norm(swing.state.rope.velocity[:, -1], axis=-1)
    handle_speed = jnp.linalg.norm(swing.state.rope.velocity[:, HANDLE], axis=-1)
    print(f'peak tip speed: {float(jnp.max(tip_speed)):.1f} m/s, '
          f'peak handle speed: {float(jnp.max(handle_speed)):.1f} m/s')
    from examples.whip.plots import plot_swing

    print(f'plot: {plot_swing(swing, command)}')


if __name__ == '__main__':
    main()
