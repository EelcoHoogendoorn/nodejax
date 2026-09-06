"""The whip plant at the example's dimensions and clocks, its figures, and
an open-loop swing.

A motor with a coarse encoder turns the handle of a rope. Under a constant
velocity command the handle spins up, the rope trails and then overtakes
it, and the tip goes several times faster than the handle. The figure shows
the rope over time, the tip speed, the controller's velocity estimate
against the true handle velocity, and the torque it took. The other
figures here, the rollout animation, the actuator readouts, and the
training curves, serve the learners.
"""

import math

import jax
import jax.numpy as jnp

from nodejax import Node, Struct, batch, scan, split_aux
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


# The clocks. Actuator and rope meet at the control tick, and both step
# once per tick: the motor's currents advance implicitly, stable at any
# tick, and the rope steps coarsely and a little compliant, which is fine
# for the learner. Were one side refined by an integer factor, it would
# substep inside the tick while the other holds, never both side by side.
# Above the tick the policy holds one command for N_CONTROL_TICKS.
DT = 1e-3  # the control tick, one actuator tick and one rope step
N_ROPE_SUBSTEPS = 1  # rope steps per control tick, under the held torque
N_CONTROL_TICKS = 10  # control ticks per policy step, under one held command
PLANT_DT = DT * N_CONTROL_TICKS
LINK_LENGTHS = (0.3, 0.3)  # upper arm and forearm, shoulder to elbow to handle
N_LINKS = len(LINK_LENGTHS)
HANDLE = N_LINKS  # the particle at the last link's end, where the rope starts
SEGMENT = 0.15
N_ROPE = 8
ROTOR_INERTIA = 0.05
ROPE_MASS = 0.05
N_SOLVER_PASSES = 4
# Bending compliance, geometric from grip to tip. The constraint's own term is about 3e3 on
# these particles, so 1e4 is a quarter strength at the grip and 1e6 is free at the tip.
BEND_COMPLIANCE_HANDLE = 1e4
BEND_COMPLIANCE_TIP = 1e6
# Below the pivot: the arm can work under it but not hang, and the rope cannot swing through.
FLOOR_HEIGHT = -0.4
ENCODER_RESOLUTION = 256.0
OBSERVER_TAU_POS = 2.0  # ticks of position filtering
# Ticks of velocity filtering: the quantized encoder's trade between noise and lag.
OBSERVER_TAU_VEL = 5.0
# Nm per rad/s of velocity error; higher and the elbow loop chatters on the coarse encoder.
VELOCITY_GAIN = 2.5
# A loop that sits on the torque clamp until within 12 rad/s of the setpoint, for sensing with
# no noise to amplify.
VELOCITY_GAIN_STIFF = 10.0
VELOCITY_KI = 0.0  # the velocity loop is proportional
CURRENT_KP = 0.5
CURRENT_KI = 500.0
CURRENT_FILTER_TAU = 2e-3  # seconds, the current sensor's smoothing
BUS_FILTER_TAU = 1e-2  # seconds, the bus estimate's smoothing
# Joules in the one pack both joints draw on: a hard stroke or two beyond an episode.
BATTERY_CAPACITY = 1200.0
BUS_VOLTAGE = 48.0
TORQUE_LIMIT = 120.0  # the stack's current limit times its torque constant, for the ideal actuator
MOTOR_KT = 1.2
TARGET = (0.0, 1.2)
PROXIMITY = 0.2
HALF_CREDIT_RADIUS = PROXIMITY * math.sqrt(math.log(2.0))  # where closeness is one half
HIT_ENERGY = 10.0  # joules, the tip at 20 m/s: the unit the cost counts hits in
HIT_POWER = 1.0  # the cost is linear in the tip's energy


def whip_at_rest(link_lengths: tuple = LINK_LENGTHS) -> Struct:
    """The example's rope on an arm of ``link_lengths``, straight and still
    along the x axis."""
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
    link_lengths: tuple = LINK_LENGTHS,
) -> Node:
    """The whip on its arm of ``link_lengths`` at the policy's clock: one
    call holds a command per joint for ``N_CONTROL_TICKS`` ticks of the
    actuators, each ticking the rope ``N_ROPE_SUBSTEPS`` times, and pays the
    cost of every tick. The joints share one actuator definition, batched
    over them, and one battery: the stack, or with ``ideal_actuator`` the
    plain torque source with a velocity loop that is the ablation against
    it. ``velocity_gain`` is the loop's, in Nm per rad/s; the default suits
    the quantized encoder, whose noise a stiffer loop turns into torque."""
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
    plant: Node,
    start: Struct,
    command: jax.Array,
    rng: jax.Array,
) -> Struct:
    """Scan the plant from ``start``, one of its own, under a command
    sequence shaped (time, joint).

    The result holds the state trajectory, the torques, the cost, and the
    true joint velocities beside the controllers' estimates of them."""
    n_steps = command.shape[0]
    stepping = plant.parameterize()
    state = stepping.init(rng=rng, start=start)
    program = scan(plant, n=n_steps).parameterize().bind(state=state)
    _, outputs = jax.jit(program.apply)(command=command, disturbance=jnp.zeros((n_steps,)))
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
