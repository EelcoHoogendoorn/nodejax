"""Couples a two-link FOC-actuated arm to a PBD particle rope with a shared `Battery` pack."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode, Composite, Leaf, Node, PNode, PSNode, Struct, Wrapper, batch, iterated, node, serial, tile,
)
from nodejax.control import EMA, PID
from examples.actuator import (
    ActuatorStack,
    CurrentController,
    CurrentSensor,
    DeratingThermal,
    Electrical,
    Encoder,
    FET,
    ModelEstimator,
    Noisy,
    Observer,
    VelocityCommand,
    foc_current_model,
)
from examples.actuator.utils import wrap as angle_wrap
from examples.pbd import (
    Constraint,
    FloorConstraint,
    ParticleBend,
    ParticleDistance,
    gauss_seidel,
    particle,
    pbd_step,
)


PIVOT = 0


def whip_particles(
    n_rope: int,
    *,
    link_lengths: tuple[float, ...],
    segment: float,
    rotor_inertia: float,
    rope_mass: float,
) -> Struct:
    """Constructs the initial `Particle` chain along the `+x` axis: pinned pivot, link endpoints carrying `rotor_inertia`, and `n_rope` rope particles."""
    lengths = jnp.asarray(link_lengths, dtype=jnp.float32)
    arm_reach = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(lengths)))
    reach = jnp.concatenate((arm_reach, arm_reach[-1] + segment * jnp.arange(1, n_rope + 1)))
    position = jnp.stack((reach, jnp.zeros_like(reach)), axis=-1)
    inverse_mass = jnp.concatenate((
        jnp.zeros((1,)),
        lengths**2 / rotor_inertia,
        jnp.full((n_rope,), 1.0 / rope_mass),
    ))
    return particle(
        position=position,
        velocity=jnp.zeros_like(position),
        inverse_mass=inverse_mass,
    )


def rotated(particles: Struct, angle: float | jax.Array) -> Struct:
    """Rotates `particles` counter-clockwise by `angle` radians about the origin, broadcasting leading batch axes of `angle`."""
    angle = jnp.asarray(angle, dtype=jnp.float32)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.stack((jnp.stack((cos, -sin), axis=-1), jnp.stack((sin, cos), axis=-1)), axis=-2)
    position = jnp.einsum('...ij,nj->...ni', rotation, particles.position)
    velocity = jnp.einsum('...ij,nj->...ni', rotation, particles.velocity)
    return particles.replace(
        position=position,
        velocity=velocity,
        inverse_mass=jnp.broadcast_to(particles.inverse_mass, position.shape[:-1]),
    )


def whip_constraints(
    n_rope: int,
    *,
    link_lengths: tuple[float, ...],
    segment: float,
    bend_compliance_handle: float,
    bend_compliance_tip: float,
) -> Struct:
    """Builds `Struct(distance, bending)` constraint parameters with rigid arm links and geometrically tapered rope bending compliance."""
    lengths = jnp.asarray(link_lengths, dtype=jnp.float32)
    n_links = lengths.shape[0]
    n_points = n_links + 1 + n_rope
    pairs = jnp.stack((jnp.arange(n_points - 1), jnp.arange(1, n_points)), axis=-1)
    rest_length = jnp.concatenate((lengths, jnp.full((n_rope,), segment)))
    distance = Struct(
        index=pairs,
        constraint=Struct(
            rest_length=rest_length,
            compliance=jnp.zeros((n_points - 1,)),
        ),
    )
    first = jnp.arange(n_links - 1, n_points - 2)
    triples = jnp.stack((first, first + 1, first + 2), axis=-1)
    taper = jnp.linspace(0.0, 1.0, first.shape[0])
    bending = Struct(
        index=triples,
        constraint=Struct(
            rest_angle=jnp.zeros((first.shape[0],)),
            compliance=(
                bend_compliance_handle * (bend_compliance_tip / bend_compliance_handle) ** taper),
        ),
    )
    return Struct(distance=distance, bending=bending)


def link_motion(particles: Struct, link_lengths: jax.Array) -> Struct:
    """Computes world-frame link angles and angular rates of shape `(..., n_links)` from link-endpoint particle positions and velocities."""
    n_links = link_lengths.shape[0]
    start = particles.position[..., PIVOT:PIVOT + n_links, :]
    end = particles.position[..., PIVOT + 1:PIVOT + n_links + 1, :]
    span = end - start
    relative = (
        particles.velocity[..., PIVOT + 1:PIVOT + n_links + 1, :]
        - particles.velocity[..., PIVOT:PIVOT + n_links, :]
    )
    angle = jnp.arctan2(span[..., 1], span[..., 0])
    rate = (span[..., 0] * relative[..., 1] - span[..., 1] * relative[..., 0]) / link_lengths**2
    return Struct(angle=angle, rate=rate)


def joint_mechanical(particles: Struct, link_lengths: jax.Array) -> Struct:
    """Computes relative joint angles and angular velocities `Struct(position, velocity)` of shape `(..., n_links)` from `particles`."""
    link = link_motion(particles, link_lengths)
    previous_angle = jnp.concatenate(
        (jnp.zeros_like(link.angle[..., :1]), link.angle[..., :-1]), axis=-1)
    previous_rate = jnp.concatenate(
        (jnp.zeros_like(link.rate[..., :1]), link.rate[..., :-1]), axis=-1)
    return Struct(
        position=angle_wrap(link.angle - previous_angle),
        velocity=link.rate - previous_rate,
    )


def joint_forces(particles: Struct, torques: jax.Array, link_lengths: jax.Array) -> jax.Array:
    """Converts joint `torques` of shape `(n_links,)` into equal-and-opposite force couples on the arm particles."""
    n_links = link_lengths.shape[0]
    start = particles.position[PIVOT:PIVOT + n_links]
    end = particles.position[PIVOT + 1:PIVOT + n_links + 1]
    span = end - start
    normal = jnp.stack((-span[:, 1], span[:, 0]), axis=-1) / link_lengths[:, None]
    force = jnp.zeros_like(particles.position)
    for joint in range(n_links):
        couple = torques[joint] / link_lengths[joint] * normal[joint]
        force = force.at[PIVOT + joint + 1].add(couple).at[PIVOT + joint].add(-couple)
        if joint > 0:
            reaction = torques[joint] / link_lengths[joint - 1] * normal[joint - 1]
            force = force.at[PIVOT + joint].add(-reaction).at[PIVOT + joint - 1].add(reaction)
    return force


@node
def Arm(link_lengths: tuple[float, ...]) -> PNode:
    """Maps `(particles, torques)` to particle forces and exposes relative joint mechanics via `.mechanical(particles)`."""
    lengths = jnp.asarray(link_lengths, dtype=jnp.float32)

    def apply(particles: Struct, torques: jax.Array) -> jax.Array:
        return joint_forces(particles, torques, lengths)

    def mechanical(particles: Struct) -> Struct:
        return joint_mechanical(particles, lengths)

    return Leaf(apply, methods={'mechanical': mechanical})


@node
def HitCost(proximity: float, hit_energy: float, hit_power: float) -> PNode:
    """Prices one rope step as negated Gaussian-weighted tip kinetic energy `-((tip_energy / hit_energy) ** hit_power) * closeness`."""
    def apply(particles: Struct, target: jax.Array) -> jax.Array:
        tip_energy = 0.5 * jnp.sum(particles.velocity[-1] ** 2) / particles.inverse_mass[-1]
        closeness = jnp.exp(-jnp.sum((particles.position[-1] - target) ** 2) / proximity**2)
        return -((tip_energy / hit_energy) ** hit_power) * closeness

    return Leaf(apply)


@node
def IdealActuator() -> Node:
    """Models a memoryless `tanh`-saturated velocity-loop torque source on exact joint velocities for actuator-stack ablations."""
    def param(velocity_gain: float, torque_limit: float, kt: float) -> Struct:
        return Struct(velocity_gain=velocity_gain, torque_limit=torque_limit, kt=kt)

    def init(input: Struct) -> Struct:
        return Struct(
            position=input.mechanical.position,
            velocity=input.mechanical.velocity,
            torque=jnp.zeros_like(input.command),
        )

    def apply(param, state, mechanical: Struct, command: jax.Array, bus_voltage: jax.Array) -> tuple[Struct, Struct]:
        torque = param.torque_limit * jnp.tanh(
            param.velocity_gain * (command - mechanical.velocity) / param.torque_limit)
        advanced = Struct(position=mechanical.position, velocity=mechanical.velocity, torque=torque)
        return advanced, Struct(torque=torque, power=jnp.zeros_like(torque))

    def observe(param, state) -> Struct:
        return Struct(
            angle=state.position, velocity=state.velocity, current=state.torque / param.kt)

    return Leaf(apply, param=param, init=init, methods={'observe': observe})


@node
def Target() -> PNode:
    """Holds a fixed episode target vector in plant state so cost, observation, critic, and planner read it from one place."""
    def init(input: jax.Array) -> jax.Array:
        return input

    def apply(state) -> tuple[jax.Array, jax.Array]:
        return state, state

    return Leaf(apply, init=init)


@node
def Whip(
    actuator: BaseNode,
    battery: BaseNode,
    rope: PSNode,
    arm: BaseNode,
    cost: BaseNode,
    *,
    target: tuple[float, ...],
    tip_sensor: bool = False,
) -> Node | PNode:
    """Steps the coupled arm, batched joint `actuator`, shared `battery`, and PBD `rope` plant for one control tick.

    Reads relative joint mechanics from `rope.state`, steps `actuator` at `battery.voltage()`, discharges `battery` by summed joint power, applies joint torques plus `disturbance` to `rope`, and evaluates `cost` against `target`. Exposes `observe`, `start`, and `mechanical` methods.
    """
    target_array = jnp.asarray(target, dtype=jnp.float32)
    members = Composite(
        actuator=actuator, battery=battery, rope=rope, arm=arm, cost=cost,
        target=Target().bind(state=target_array))
    default_start = Struct(rope=rope.state, target=target_array)

    def apply(self, command: jax.Array, disturbance: jax.Array) -> Struct:
        mechanical = self.arm.mechanical(self.rope.state)
        bus_voltage = jnp.full_like(mechanical.velocity, self.battery.voltage())
        driven = self.actuator(mechanical=mechanical, command=command, bus_voltage=bus_voltage)
        self.battery(jnp.sum(driven.power))
        particles = self.rope(self.arm(self.rope.state, driven.torque + disturbance))
        return Struct(action=driven.torque, cost=self.cost(particles, self.target.state))

    def init(self, start: Struct = default_start) -> None:
        """Initializes `rope` and `target` from `start` and warm-resets `actuator` at the initial link angles and full bus voltage."""
        self.rope.bind(state=start.rope)
        self.target.bind(state=start.target)
        mechanical = self.arm.mechanical(start.rope)
        self.actuator.reset(input=Struct(
            mechanical=mechanical,
            command=jnp.zeros_like(mechanical.velocity),
            bus_voltage=jnp.full_like(mechanical.velocity, self.battery.voltage()),
        ))

    def observe(state) -> Struct:
        """Reads controller estimates from `state.actuator`, attaches `state.target`, and appends tip kinematics when `tip_sensor=True`."""
        observation = actuator.observe(state=state.actuator).replace(target=state.target)
        if tip_sensor:
            observation = observation.replace(
                tip_position=state.rope.position[-1],
                tip_velocity=state.rope.velocity[-1],
            )
        return observation

    def start(angle, target) -> Struct:
        """Constructs an episode start `Struct(rope, target)` with the resting whip rotated to `angle`."""
        return Struct(
            rope=rotated(rope.state, angle), target=jnp.asarray(target, dtype=jnp.float32))

    def mechanical(state) -> Struct:
        """Extracts joint angles and angular velocities from `state.rope`."""
        return arm.mechanical(particles=state.rope)

    return members(
        apply, init=init, methods={'observe': observe, 'start': start, 'mechanical': mechanical})


@node
def WhipStep(tick: BaseNode, n_ticks: int) -> Node | PNode:
    """Scans `tick` across `n_ticks` subticks under held `(command, disturbance)` and sums tick costs."""
    def apply(self, command: jax.Array, disturbance: jax.Array) -> Struct:
        ticks = self.tick.scan(
            command=tile(command, n_ticks), disturbance=tile(disturbance, n_ticks))
        return Struct(
            state=self.tick.state,
            action=ticks.action[-1],
            cost=jnp.sum(ticks.cost),
        )

    return Wrapper(tick=tick)(apply)


def whip_actuator(
    dt: float,
    *,
    encoder_resolution: float,
    observer_tau_pos: float,
    observer_tau_vel: float,
    velocity_kp: float,
    velocity_ki: float,
    current_kp: float,
    current_ki: float,
    current_filter_tau: float,
    bus_filter_tau: float,
    exact_encoder: bool = False,
) -> Node:
    """Builds a velocity-commanded `ActuatorStack` with optional `Encoder` quantization bypass (`exact_encoder=True`)."""
    motor = Electrical(dt)(
        resistance=0.24,
        inductance_d=2e-4,
        inductance_q=3e-4,
        kt=1.2,
        pole_pairs=16.0,
        slots=36.0,
        cogging=0.8,
    )
    observer = Observer(dt)(tau_pos=observer_tau_pos, tau_vel=observer_tau_vel)
    return ActuatorStack(
        mechanical_est=(
            serial(observer=observer) if exact_encoder else
            Encoder()(resolution=encoder_resolution) >> observer
        ),
        command_ctrl=VelocityCommand(PID(dt)(kp=velocity_kp, ki=velocity_ki)),
        current_ctrl=CurrentController(
            dt,
            motor=motor,
            estimator=ModelEstimator(
                dt,
                filter=CurrentSensor() >> EMA(dt, warm=True)(tau=current_filter_tau),
                model_fn=foc_current_model(dt),
            ),
            controller=PID(dt)(kp=current_kp, ki=current_ki),
            fets=FET(dt)(r_th=2.0, c_th=5.0, limit=80.0, r_dson=0.02),
            bus_est=Noisy()(noise_std=0.2) >> EMA(dt, warm=True)(tau=bus_filter_tau),
        ).parameterize(
            ff=Struct(r=1.0, bemf=1.0, l=0.0),
            limit=Struct(limit=100.0),
        ),
        motor=motor,
        motor_thermal=DeratingThermal(dt)(r_th=1.5, c_th=40.0, limit=100.0),
    )


def whip_rope(
    constraints: Struct,
    *,
    dt: float,
    n_substeps: int,
    floor_height: float,
    n_solver_passes: int,
) -> PNode:
    """Builds the `n_substeps`-iterated cyclic PBD rope solver projecting distance, bending, and `floor_height` constraints."""
    solve = serial(
        distance=gauss_seidel(constraints.distance, Constraint(ParticleDistance())),
        bending=gauss_seidel(constraints.bending, Constraint(ParticleBend())),
        floor=batch(FloorConstraint(floor_height)),
    )
    step = pbd_step(solve, n_solver_passes=n_solver_passes, dt=dt / n_substeps)
    return iterated(step, n=n_substeps)
