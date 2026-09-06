"""A whip on an arm: the FOC actuator stack drives a two-link arm holding a rope.

The rope is the particle world from ``examples.pbd``, the motor is the
actuator stack from ``examples.actuator``, and ``Whip`` is what joins them.
The arm is part of the rope's particle record: a pivot, one particle at the
end of each rigid link, and the rope from the last link's end, the handle.
Each link is driven by a joint motor, the first against the world and each
next one against the link before it, and the joints' actuators are one
actuator batched over the joints. A joint's angle and rate are read off its
link, and its torque goes back as a couple on that link, so the rope
integrates the rotors and the load on every motor comes back through the
constraint projection.

The particle layout is fixed: index 0 is the pivot, indices 1 to
``n_links`` the link ends, the last of them the handle, and the rest the
rope from handle to tip.

The policy commands the joints and observes only what their controllers
know, per joint the observer's angle and velocity estimate and the
estimated quadrature current. The rope is felt through the load.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    Composite, Leaf, Node, PSNode, Struct, Wrapper, batch, iterated, node, serial, tile,
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
    link_lengths: tuple,
    segment: float,
    rotor_inertia: float,
    rope_mass: float,
) -> Struct:
    """The particle record of a whip at rest along the x axis: pivot, one
    link end per entry of ``link_lengths``, and ``n_rope`` rope particles of
    ``rope_mass`` each, ``segment`` apart. Each link end carries its joint's
    rotor inertia as a point mass."""
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
    """The particle record turned by ``angle`` radians about the origin.
    Leading axes of ``angle`` become leading axes of the record, one whip
    each."""
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
    link_lengths: tuple,
    segment: float,
    bend_compliance_handle: float,
    bend_compliance_tip: float,
) -> Struct:
    """The constraint rows of a whip: ``distance`` rows joining pivot, link
    ends, and rope in a chain, and ``bending`` rows on every triple from the
    handle's grip to the tip, their compliance growing geometrically from
    the handle's to the tip's. The joints between links carry no bending
    row: they are free, and their motors hold them."""
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
    """Each link's angle in the world and its angular rate, read off the
    particles at its ends, shaped (link,). Leading axes of the record, a
    trajectory or a batch of whips, carry through."""
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
    """The rotors' angles and angular velocities, shaped (joint,): the first
    joint turns its link against the world, each next one against the link
    before it. Leading axes of the record carry through, as in ``link_motion``."""
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
    """Joint torques, shaped (joint,), as forces on one particle record:
    each torque is a couple on the link it drives and the opposite couple
    on the link before it, the pivot absorbing what falls on it."""
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
def Arm(link_lengths: tuple) -> Node:
    """The rigid links from the pivot to the handle, as the rope's particle
    record carries them. Applied to the record and the joint torques it
    gives the forces those torques are on the particles; ``mechanical``
    reads each joint's angle and velocity off its link."""
    lengths = jnp.asarray(link_lengths, dtype=jnp.float32)

    def apply(particles, torques):
        return joint_forces(particles, torques, lengths)

    def mechanical(particles) -> Struct:
        return joint_mechanical(particles, lengths)

    return Leaf(apply, methods={'mechanical': mechanical})


@node
def HitCost(proximity: float, hit_energy: float, hit_power: float) -> Node:
    """The cost of one rope step given the target: the tip's kinetic
    energy in units of ``hit_energy`` raised to ``hit_power``, weighted by
    a Gaussian of width ``proximity`` around the target, negated. A faster
    pass earns more and not merely a quicker one, and above one the power
    makes one hard hit outrank many soft ones."""
    def apply(particles, target):
        tip_energy = 0.5 * jnp.sum(particles.velocity[-1] ** 2) / particles.inverse_mass[-1]
        closeness = jnp.exp(-jnp.sum((particles.position[-1] - target) ** 2) / proximity**2)
        return -((tip_energy / hit_energy) ** hit_power) * closeness

    return Leaf(apply)


@node
def IdealActuator() -> Node:
    """A torque source with a velocity loop on the true joint velocity and
    nothing else: no encoder, no currents, no heat, a smooth torque
    ceiling, and no draw on the bus it is handed. The ablation against the
    actuator stack: what a policy trained through this does not suffer is
    the stack's doing. Its gain, torque ceiling, and torque constant are
    parameters, as the stack's are. It keeps the mechanical state it last
    saw and its last torque, which ``observe`` reports in the stack's
    terms, the torque as the current it would take.
    """
    def param(velocity_gain: float, torque_limit: float, kt: float) -> Struct:
        return Struct(velocity_gain=velocity_gain, torque_limit=torque_limit, kt=kt)

    def init(input):
        return Struct(
            position=input.mechanical.position,
            velocity=input.mechanical.velocity,
            torque=jnp.zeros_like(input.command),
        )

    def apply(param, state, mechanical, command, bus_voltage):
        torque = param.torque_limit * jnp.tanh(
            param.velocity_gain * (command - mechanical.velocity) / param.torque_limit)
        advanced = Struct(position=mechanical.position, velocity=mechanical.velocity, torque=torque)
        return advanced, Struct(torque=torque, power=jnp.zeros_like(torque))

    def observe(param, state) -> Struct:
        return Struct(
            angle=state.position, velocity=state.velocity, current=state.torque / param.kt)

    return Leaf(apply, param=param, init=init, methods={'observe': observe})


@node
def Target() -> Node:
    """The point the whip is to hit this episode, held as state beside the
    rope so the cost, the observation, a critic, and a planner all find it
    where the plant is. Set at the start and never advanced."""
    def init(input):
        return input

    def apply(state):
        return state, state

    return Leaf(apply, init=init)


@node
def Whip(
    actuator: Node,
    battery: Node,
    rope: PSNode,
    arm: Node,
    cost: Node,
    *,
    target: tuple,
    tip_sensor: bool = False,
) -> Node:
    """The plant: motors turning the joints of an arm that holds a rope,
    fed by one battery. ``start`` makes a start for the plant, the rope at
    rest at an angle and a target; ``mechanical`` reads the joints off a
    state.

    ``command`` is a velocity setpoint per joint, in rad/s and unbounded:
    whatever the actuator cannot follow, its own limits clip. It goes to
    the actuator, which is batched over the joints and fed the battery's
    voltage; the powers the joints draw are summed back into the battery,
    so the pack is shared. ``disturbance`` is a torque added at every
    joint. ``arm`` reads the joints off the rope's record and puts the
    torques back as couples; ``cost`` prices each rope step against the
    target. Effort is not charged: whatever caution the policy learns comes
    from the pack it drains and the thermal rollback of the motor. The
    state is the members': the actuator's, the battery's, the rope's, and
    the target's. ``observe`` is the controllers' own view of it with the target, plus
    the tip's position and velocity when the plant has a ``tip_sensor``,
    the privileged variant for asking whether the rope's invisibility is
    what limits a policy. The start is a record of ``rope`` and ``target``;
    ``rope`` arrives state bound, at rest, and with the static ``target``
    it is the start unless init is given another, and where
    parameterization reads the state's shape.
    """
    target = jnp.asarray(target, dtype=jnp.float32)
    members = Composite(
        actuator=actuator, battery=battery, rope=rope, arm=arm, cost=cost,
        target=Target().bind(state=target))
    default_start = Struct(rope=rope.state, target=target)

    def apply(self, command, disturbance):
        mechanical = self.arm.mechanical(self.rope.state)
        bus_voltage = jnp.full_like(mechanical.velocity, self.battery.voltage())
        driven = self.actuator(mechanical=mechanical, command=command, bus_voltage=bus_voltage)
        self.battery(jnp.sum(driven.power))
        particles = self.rope(self.arm(self.rope.state, driven.torque + disturbance))
        return Struct(action=driven.torque, cost=self.cost(particles, self.target.state))

    def init(self, start=default_start):
        """Start from ``start``, a record of the rope and the target, with
        every joint's actuator primed at rest on its link; the pack starts
        full on its own."""
        self.rope.bind(state=start.rope)
        self.target.bind(state=start.target)
        mechanical = self.arm.mechanical(start.rope)
        self.actuator.reset(input=Struct(
            mechanical=mechanical,
            command=jnp.zeros_like(mechanical.velocity),
            bus_voltage=jnp.full_like(mechanical.velocity, self.battery.voltage()),
        ))

    def observe(state) -> Struct:
        """The actuator's own readout, per joint, and the target; and the
        tip, if the plant has a sensor on it."""
        observation = actuator.observe(state=state.actuator).replace(target=state.target)
        if tip_sensor:
            observation = observation.replace(
                tip_position=state.rope.position[-1],
                tip_velocity=state.rope.velocity[-1],
            )
        return observation

    def start(angle, target) -> Struct:
        """A start: the rope at rest, turned to ``angle`` radians, and the
        target. Arrays of angles, shaped (world,), and targets, shaped
        (world, 2), give a start per world."""
        return Struct(
            rope=rotated(rope.state, angle), target=jnp.asarray(target, dtype=jnp.float32))

    def mechanical(state) -> Struct:
        """The joints' angles and velocities read off the rope of ``state``;
        leading axes, a trajectory or worlds, carry through."""
        return arm.mechanical(particles=state.rope)

    return members(
        apply, init=init, methods={'observe': observe, 'start': start, 'mechanical': mechanical})


@node
def WhipStep(tick: Node, n_ticks: int) -> Node:
    """The whip at the policy's clock: ``tick`` scanned ``n_ticks`` times
    under one held command and disturbance. The cost is the sum over the
    ticks, so a crack that crosses the target between two policy steps is
    paid in full, and the action is the last tick's torque."""
    def apply(self, command, disturbance):
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
    """The stock actuator stack under a velocity command, with an encoder of
    ``encoder_resolution`` counts per revolution, an observer filtering
    position over ``observer_tau_pos`` ticks and velocity over
    ``observer_tau_vel``, a velocity loop, a current loop, and the current
    and bus sensor filters, each at the value given. A coarse encoder needs
    a longer velocity average and the velocity loop inherits the lag; that
    is the actuator the policy has to learn around. With ``exact_encoder``
    there is no encoder at all and the observer reads the true angle: the
    ablation that removes the rounding from the signal and from the
    gradient alike, since a round has no gradient at any resolution. The
    stack is fed its bus voltage and reports its power; the pack lives with
    the plant."""
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
        # Either way the estimator is a pipe ending in its observer, which is
        # where the stack's readout looks.
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
) -> Node:
    """The rope world at the control tick: the chain's ``distance`` rows and
    its ``bending`` rows from ``constraints``, and a floor at
    ``floor_height`` that no particle passes below, projected in that
    order, cyclic over its particles, ``n_substeps`` steps of
    ``dt / n_substeps`` under the force the actuator holds for the tick.
    The floor is what keeps the arm from windmilling."""
    solve = serial(
        distance=gauss_seidel(constraints.distance, Constraint(ParticleDistance())),
        bending=gauss_seidel(constraints.bending, Constraint(ParticleBend())),
        floor=batch(FloorConstraint(floor_height)),
    )
    step = pbd_step(solve, n_solver_passes=n_solver_passes, dt=dt / n_substeps)
    return iterated(step, n=n_substeps)
