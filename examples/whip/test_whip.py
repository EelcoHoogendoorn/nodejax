"""The whip plant: primed from one key, ticked, and swung open loop."""

import jax
import jax.numpy as jnp

from nodejax import Struct, split_aux
from examples.whip.plant import PIVOT, joint_forces, joint_mechanical, rotated
from examples.whip.whip import (
    HANDLE,
    LINK_LENGTHS,
    N_LINKS,
    N_ROPE,
    TARGET,
    open_loop_swing,
    whip_at_rest,
    whip_plant,
)


def whip_start(angle: float | jax.Array = 0.0, target: tuple | jax.Array = TARGET) -> Struct:
    """A start of the default plant."""
    return whip_plant().parameterize().start(angle=angle, target=target)


def shoulder_command(n_steps: int, value: float) -> jax.Array:
    """A command sequence that drives the shoulder and holds the elbow."""
    return jnp.zeros((n_steps, N_LINKS)).at[:, 0].set(value)


def shoulder_strokes(n_steps: int, value: float, period: int) -> jax.Array:
    """Shoulder commands reversing every ``period`` steps, the elbow held."""
    sign = jnp.where((jnp.arange(n_steps) // period) % 2 == 0, 1.0, -1.0)
    return jnp.zeros((n_steps, N_LINKS)).at[:, 0].set(value * sign)


def test_joint_mechanical_reads_the_rotors_off_the_links() -> None:
    particles = rotated(whip_at_rest(), 0.5)
    elbow_velocity = 3.0 * LINK_LENGTHS[0] * jnp.array((-jnp.sin(0.5), jnp.cos(0.5)))
    spun = particles.replace(
        velocity=particles.velocity.at[1].set(elbow_velocity).at[HANDLE].set(2.0 * elbow_velocity),
    )
    mechanical = joint_mechanical(spun, jnp.asarray(LINK_LENGTHS))
    assert jnp.allclose(mechanical.position, jnp.array((0.5, 0.0)), atol=1e-6)
    # The arm rotates rigidly at 3 rad/s: the shoulder turns, the elbow does not.
    assert jnp.allclose(mechanical.velocity, jnp.array((3.0, 0.0)), atol=1e-5)


def test_joint_forces_are_couples_on_the_links() -> None:
    particles = rotated(whip_at_rest(), 0.5)
    force = joint_forces(particles, jnp.array((0.0, 6.0)), jnp.asarray(LINK_LENGTHS))
    elbow, handle = particles.position[1], particles.position[HANDLE]
    moment_about_elbow = (handle - elbow)[0] * force[HANDLE, 1] - (handle - elbow)[1] * force[HANDLE, 0]
    assert jnp.allclose(moment_about_elbow, 6.0)
    # The couple on the forearm and the reaction on the upper arm sum to nothing on the free particles.
    assert jnp.allclose(jnp.sum(force[1:], axis=0) + force[PIVOT], 0.0, atol=1e-5)
    assert jnp.all(force[HANDLE + 1:] == 0.0)


def test_plant_primes_from_one_key_and_the_rope() -> None:
    plant = whip_plant().parameterize()
    state = plant.init(rng=jax.random.PRNGKey(0), start=whip_start())
    observation = plant.observe(state=state)

    assert state.rope.position.shape == (N_LINKS + 1 + N_ROPE, 2)
    assert state.target.shape == (2,) and observation.target.shape == (2,)
    assert observation.angle.shape == (N_LINKS,)
    assert jnp.allclose(observation.angle, 0.0)
    assert jnp.allclose(observation.velocity, 0.0)
    assert jnp.allclose(state.actuator.current_ctrl.bus_est.ema, 48.0, atol=1.0)


def test_one_step_keeps_the_links_rigid_and_turns_the_shoulder() -> None:
    plant = whip_plant().parameterize()
    state = plant.init(rng=jax.random.PRNGKey(0), start=whip_start())
    started = plant.bind(state=state)

    advanced, output = started.apply(command=jnp.array((40.0, 0.0)), disturbance=jnp.zeros(()))
    output, _ = split_aux(output)

    positions = advanced.state.rope.position
    links = jnp.linalg.norm(positions[1:N_LINKS + 1] - positions[:N_LINKS], axis=-1)
    assert jnp.allclose(links, jnp.asarray(LINK_LENGTHS), atol=1e-4)
    assert jnp.all(positions[PIVOT] == 0.0)
    assert output.action.shape == (N_LINKS,)
    assert jnp.all(jnp.isfinite(output.action))
    assert joint_mechanical(advanced.state.rope, jnp.asarray(LINK_LENGTHS)).velocity[0] > 0.0
    assert output.cost.shape == ()
    assert jax.tree.structure(advanced.state) == jax.tree.structure(state)


def test_open_loop_strokes_swing_the_shoulder_and_the_tip_outruns_the_handle() -> None:
    n_steps = 100
    command = shoulder_strokes(n_steps, 30.0, period=50)
    swing = open_loop_swing(whip_plant(), whip_start(), command, jax.random.PRNGKey(1))

    tip_speed = jnp.linalg.norm(swing.state.rope.velocity[:, -1], axis=-1)
    handle_speed = jnp.linalg.norm(swing.state.rope.velocity[:, HANDLE], axis=-1)
    assert jnp.all(jnp.isfinite(swing.state.rope.position))
    assert jnp.max(swing.joint_velocity[:, 0]) > 10.0
    assert jnp.min(swing.joint_velocity[:, 0]) < -8.0  # the back stroke meets the floor sooner than the forward one
    assert jnp.max(tip_speed) > 1.5 * jnp.max(handle_speed)
    # The controllers' estimates follow the true joint velocities, coarse encoders and all.
    assert jnp.all(jnp.abs(jnp.mean(swing.estimated_velocity - swing.joint_velocity, axis=0)) < 3.0)


def test_a_coarser_encoder_gives_a_rougher_estimate() -> None:
    n_steps = 80
    command = shoulder_strokes(n_steps, 30.0, period=40)
    settled = slice(n_steps // 2, None)

    def roughness(resolution: float) -> jax.Array:
        swing = open_loop_swing(whip_plant(resolution), whip_start(), command, jax.random.PRNGKey(1))
        return jnp.std(swing.estimated_velocity[settled, 0] - swing.joint_velocity[settled, 0])

    assert roughness(32.0) > roughness(2048.0)


def test_priming_from_any_angle_leaves_the_observers_at_rest() -> None:
    plant = whip_plant().parameterize()
    for angle in (0.0, 1.5, 3.1):
        state = plant.init(rng=jax.random.PRNGKey(0), start=whip_start(angle))
        observation = plant.observe(state=state)
        assert jnp.allclose(observation.angle, jnp.array((angle, 0.0)), atol=0.05)
        assert jnp.allclose(observation.velocity, 0.0)


def test_a_policy_step_pays_the_cost_of_every_tick() -> None:
    from examples.whip.whip import N_CONTROL_TICKS

    step = whip_plant().parameterize()
    tick = step.tick.parameterize()
    state = step.init(rng=jax.random.PRNGKey(0), start=whip_start())
    command, disturbance = jnp.array((30.0, -10.0)), jnp.zeros(())

    _, (output, _) = step.bind(state=state).apply(command=command, disturbance=disturbance)

    ticking = tick.bind(state=state)
    total = 0.0
    for _ in range(N_CONTROL_TICKS):
        ticking, (tick_output, _) = ticking.apply(command=command, disturbance=disturbance)
        total = total + tick_output.cost
    assert jnp.allclose(output.cost, total)


def test_the_floor_keeps_the_arm_and_rope_above_the_pivot() -> None:
    from examples.whip.whip import FLOOR_HEIGHT

    n_steps = 80
    command = shoulder_command(n_steps, -30.0)  # swing down, into the floor
    swing = open_loop_swing(whip_plant(), whip_start(0.3), command, jax.random.PRNGKey(1))
    assert jnp.all(swing.state.rope.position[..., 1] >= FLOOR_HEIGHT - 1e-5)


def test_the_ideal_actuator_plant_primes_steps_and_reads_out_like_the_stack() -> None:
    plant = whip_plant(ideal_actuator=True).parameterize()
    assert plant.contract.init_takes_rng is False  # nothing in this plant draws
    state = plant.init(start=whip_start(0.4))
    observation = plant.observe(state=state)
    assert set(observation.__keys__) == {'angle', 'velocity', 'current', 'target'}
    assert jnp.allclose(observation.angle, jnp.array((0.4, 0.0)), atol=1e-6)

    advanced, output = plant.bind(state=state).apply(command=jnp.array((40.0, 0.0)), disturbance=jnp.zeros(()))
    output, _ = split_aux(output)
    assert output.action.shape == (N_LINKS,)
    assert joint_mechanical(advanced.state.rope, jnp.asarray(LINK_LENGTHS)).velocity[0] > 0.0


def test_the_target_is_part_of_the_start_and_of_the_cost() -> None:
    plant = whip_plant().parameterize()
    target = jnp.array((0.9, 0.9))
    state = plant.init(rng=jax.random.PRNGKey(0), start=whip_start(0.5, target))
    assert jnp.allclose(state.target, target)
    assert jnp.allclose(plant.observe(state=state).target, target)
    advanced, _ = plant.bind(state=state).apply(command=jnp.zeros((N_LINKS,)), disturbance=jnp.zeros(()))
    assert jnp.allclose(advanced.state.target, target)


def test_an_exact_encoder_reads_the_true_angle() -> None:
    plant = whip_plant(exact_encoder=True).parameterize()
    state = plant.init(rng=jax.random.PRNGKey(0), start=whip_start(0.7))
    assert jnp.allclose(plant.observe(state=state).angle, jnp.array((0.7, 0.0)), atol=1e-6)


def test_starts_and_joint_readouts_broadcast_over_leading_axes() -> None:
    angles = jnp.array((0.2, 0.9, 1.7))
    targets = jnp.array(((1.0, 0.5), (0.0, 1.2), (-0.8, 0.9)))
    plant = whip_plant().parameterize()
    batched = plant.start(angle=angles, target=targets)
    single = jax.vmap(lambda angle, target: plant.start(angle=angle, target=target))(angles, targets)
    assert jax.tree.all(jax.tree.map(jnp.allclose, batched, single))
    mechanical = joint_mechanical(batched.rope, jnp.asarray(LINK_LENGTHS))
    assert mechanical.position.shape == (3, N_LINKS)
    assert jnp.allclose(mechanical.position[:, 0], angles)
