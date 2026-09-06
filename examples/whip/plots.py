"""Plots and animations of the whip: an open-loop swing, a rollout as a
GIF, the actuator's readouts along a policy rollout, and training curves
against wall time. Scripting over the example's data; nothing here is a
Node."""

import os

import jax
import jax.numpy as jnp

from nodejax import Struct
from examples.whip.whip import (
    BUS_VOLTAGE,
    ENCODER_RESOLUTION,
    HALF_CREDIT_RADIUS,
    HANDLE,
    N_LINKS,
    PLANT_DT,
)

def plot_swing(swing: Struct, command: jax.Array, filename: str = 'whip_open_loop.png') -> str:
    """Draw the rope over time and the traces of the swing."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    positions = np.asarray(swing.state.rope.position)
    velocities = np.asarray(swing.state.rope.velocity)
    n_steps = positions.shape[0]
    time = np.arange(n_steps) * PLANT_DT
    tip_speed = np.linalg.norm(velocities[:, -1], axis=-1)
    handle_speed = np.linalg.norm(velocities[:, HANDLE], axis=-1)
    joint_colors = ('k', '0.55')
    estimate_colors = ('tab:orange', 'tab:red')

    figure = plt.figure(figsize=(14.0, 8.0))
    grid = figure.add_gridspec(3, 2, width_ratios=(1.1, 1.0))

    frames = np.linspace(0, n_steps - 1, 9, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(frames)))
    axis = figure.add_subplot(grid[:, 0])
    axis.plot(
        positions[:, -1, 0], positions[:, -1, 1], ':', color='0.6', linewidth=0.8,
        label='tip trail')
    for frame, color in zip(frames, colors):
        axis.plot(positions[frame, :, 0], positions[frame, :, 1], '-o', color=color,
                  markersize=2.5, linewidth=1.3, label=f'{time[frame]:.2f} s')
    target = np.asarray(swing.state.target[0])
    axis.plot(*target, marker='x', color='crimson', markersize=10, label='target')
    axis.add_patch(plt.Circle(
        target, HALF_CREDIT_RADIUS, fill=False, color='crimson', linestyle='--', linewidth=0.8,
        label='half credit'))
    axis.set_aspect('equal')
    axis.grid(alpha=0.2)
    axis.set_title('the rope over time')
    axis.legend(fontsize=7.5, frameon=False, loc='lower left')

    speed_axis = figure.add_subplot(grid[0, 1])
    speed_axis.plot(time, tip_speed, color='k', label='tip')
    speed_axis.plot(time, handle_speed, color='0.5', label='handle')
    speed_axis.set_ylabel('speed (m/s)')
    speed_axis.set_title('the tip outruns the handle')
    speed_axis.legend(frameon=False, fontsize=8)
    speed_axis.grid(alpha=0.2)

    velocity_axis = figure.add_subplot(grid[1, 1], sharex=speed_axis)
    for joint in range(N_LINKS):
        velocity_axis.plot(
            time, np.asarray(swing.joint_velocity[:, joint]), color=joint_colors[joint],
            label=f'joint {joint} true')
        velocity_axis.plot(
            time, np.asarray(swing.estimated_velocity[:, joint]), color=estimate_colors[joint],
            linewidth=0.8, label=f'joint {joint} estimate')
        velocity_axis.plot(
            time, np.asarray(command[:, joint]), '--', color=joint_colors[joint], linewidth=0.8)
    velocity_axis.set_ylabel('joint velocity (rad/s), commands dashed')
    velocity_axis.set_title(f'what the controller sees, {ENCODER_RESOLUTION:.0f} counts per turn')
    velocity_axis.legend(frameon=False, fontsize=8)
    velocity_axis.grid(alpha=0.2)

    torque_axis = figure.add_subplot(grid[2, 1], sharex=speed_axis)
    for joint in range(N_LINKS):
        torque_axis.plot(time, np.asarray(swing.action[:, joint]), label=f'joint {joint}')
    torque_axis.legend(frameon=False, fontsize=8)
    torque_axis.set_ylabel('torque (Nm)')
    torque_axis.set_xlabel('time (s)')
    torque_axis.set_title('motor torque')
    torque_axis.grid(alpha=0.2)

    figure.suptitle('a whip on an arm: FOC actuator stacks driving a PBD rope', fontweight='bold')
    figure.tight_layout()
    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def animate_rollout(
    positions,
    target,
    filename: str,
    *,
    stride: int = 1,
    fps: int = 30,
) -> str:
    """Write a GIF of one rope trajectory, ``positions`` shaped (time,
    particle, 2), every ``stride`` steps a frame, the tip's trail so far
    drawn beneath it and ``target`` marked with the circle inside which a
    hit earns at least half credit."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    positions = np.asarray(positions)
    reach = float(np.max(np.linalg.norm(positions, axis=-1))) * 1.05
    frames = range(0, positions.shape[0], stride)

    figure, axis = plt.subplots(figsize=(6.0, 6.0))
    axis.set_xlim(-reach, reach)
    axis.set_ylim(-reach, reach)
    axis.set_aspect('equal')
    axis.grid(alpha=0.2)
    axis.plot(*np.asarray(target), marker='x', color='crimson', markersize=10)
    axis.add_patch(plt.Circle(
        np.asarray(target), HALF_CREDIT_RADIUS, fill=False, color='crimson', linestyle='--',
        linewidth=0.8))
    trail, = axis.plot([], [], ':', color='0.6', linewidth=0.8)
    rope, = axis.plot([], [], '-o', color='tab:blue', markersize=3.0, linewidth=1.5)
    clock = axis.text(0.02, 0.96, '', transform=axis.transAxes)

    def draw(frame):
        trail.set_data(positions[:frame + 1, -1, 0], positions[:frame + 1, -1, 1])
        rope.set_data(positions[frame, :, 0], positions[frame, :, 1])
        clock.set_text(f'{frame * PLANT_DT:.2f} s')
        return trail, rope, clock

    animation = FuncAnimation(figure, draw, frames=frames, blit=True)
    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output


def plot_actuator_readouts(
    evaluation: Struct,
    world: int,
    filename: str = 'whip_actuator_readouts.png',
) -> str:
    """Draw what the actuator saw and did along one closed-loop rollout:
    per joint the policy's velocity setpoint, the controller's estimate, and
    the true joint velocity, then true against estimated currents, torque
    and duty, the bus, the temperatures, and what the strokes bought, the
    tip's speed and its distance to the target. ``evaluation`` holds ``state``
    shaped (world, time), ``mechanical`` the joints read off it, and
    ``command`` and ``action`` shaped (world, time, joint)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    state = jax.tree.map(lambda value: np.asarray(value[world]), evaluation.state)
    torque = np.asarray(evaluation.action[world])
    setpoint = np.asarray(evaluation.command[world])
    actuator = state.actuator
    time = np.arange(torque.shape[0]) * PLANT_DT
    true = jax.tree.map(lambda value: np.asarray(value[world]), evaluation.mechanical)
    duty = np.hypot(actuator.current_ctrl.pwm_prev.d, actuator.current_ctrl.pwm_prev.q)
    joint_colors = ('k', '0.55')
    estimate_colors = ('tab:orange', 'tab:red')

    tip_speed = np.linalg.norm(state.rope.velocity[:-1, -1], axis=-1)
    tip_distance = np.linalg.norm(state.rope.position[:-1, -1] - state.target[0], axis=-1)

    figure, axes = plt.subplots(4, 2, figsize=(14.0, 12.0), sharex=True)

    for joint in range(N_LINKS):
        axis = axes[0, joint]
        axis.plot(time, setpoint[:, 1 - joint], color='0.7', linewidth=0.8, linestyle=':',
                  label=f"joint {1 - joint}'s setpoint, for comparison")
        axis.plot(
            time, setpoint[:, joint], color='tab:blue', linewidth=0.8, linestyle='--',
            label='policy setpoint')
        axis.plot(
            time, np.asarray(true.velocity)[:-1, joint], color=joint_colors[joint], label='true')
        axis.plot(
            time, actuator.mechanical_est.observer.velocity[:-1, joint],
            color=estimate_colors[joint], linewidth=0.8, label='estimate')
        axis.set_ylabel('joint velocity (rad/s)')
        axis.set_title(f'joint {joint}: what the policy asks and the controller sees')
        axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    for joint in range(N_LINKS):
        axis.plot(
            time, actuator.motor.q[:-1, joint], color=joint_colors[joint],
            label=f'joint {joint} true q')
        axis.plot(
            time, actuator.current_ctrl.estimator.prev.q[:-1, joint], color=estimate_colors[joint],
            linewidth=0.8, label=f'joint {joint} estimated q')
    axis.set_ylabel('current (A)')
    axis.set_title('winding currents')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    for joint in range(N_LINKS):
        axis.plot(time, torque[:, joint], label=f'joint {joint}')
    axis.set_ylabel('torque (Nm)')
    axis.set_title('motor torque')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2, 0]
    for joint in range(N_LINKS):
        axis.plot(time, duty[:-1, joint], label=f'joint {joint}')
    axis.axhline(1.0, color='crimson', linewidth=0.8, linestyle='--')
    axis.set_ylabel('duty (fraction of bus)')
    axis.set_title('voltage use, the limit dashed')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2, 1]
    axis.plot(
        time, BUS_VOLTAGE * state.battery[:-1], color='k',
        label='bus (V), one pack for both joints')
    for joint in range(N_LINKS):
        axis.plot(time, actuator.current_ctrl.bus_est.ema[:-1, joint], color=estimate_colors[joint],
                  linewidth=0.8, label=f'joint {joint} bus estimate')
    axis.set_ylabel('volts')
    axis.set_title('the shared bus')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[3, 0]
    for joint in range(N_LINKS):
        axis.plot(
            time, actuator.motor_thermal[:-1, joint], color=joint_colors[joint],
            label=f'joint {joint} windings')
    axis.set_ylabel('temperature (C)')
    axis.set_xlabel('time (s)')
    axis.set_title('thermals')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[3, 1]
    axis.plot(time, tip_speed, color='k', label='tip speed (m/s)')
    axis.set_ylabel('m/s')
    axis.set_xlabel('time (s)')
    axis.set_title('what the strokes bought: the tip')
    distance_axis = axis.twinx()
    distance_axis.plot(
        time, tip_distance, color='tab:green', linewidth=0.8, label='distance to target (m)')
    distance_axis.axhline(HALF_CREDIT_RADIUS, color='tab:green', linewidth=0.8, linestyle=':',
                          label='half credit radius')
    distance_axis.set_ylabel('m')
    distance_axis.set_ylim(bottom=0.0)
    lines, labels = axis.get_legend_handles_labels()
    distance_lines, distance_labels = distance_axis.get_legend_handles_labels()
    axis.legend(lines + distance_lines, labels + distance_labels, frameon=False, fontsize=8)

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2)
    figure.suptitle(
        'the actuator along a closed-loop rollout of the trained policy', fontweight='bold')
    figure.tight_layout()
    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_training_curves(runs: dict, filename: str) -> str:
    """Draw the mean cost per step over training against wall time, one
    curve per run: ``runs`` maps a label to a record of ``curve``, a vector
    over training rounds, and ``seconds``, the wall time of the whole run.
    The rounds are spread evenly along that time; a run is one compiled
    program, so compilation is inside it and no finer clock exists."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for name, run in runs.items():
        values = np.asarray(run.curve).reshape(-1)
        rounds = np.arange(1, values.shape[0] + 1)
        axis.plot(
            rounds / values.shape[0] * run.seconds, values, label=f'{name}, {run.seconds:.0f} s')
    axis.set_xlabel('wall time (s), rounds spread evenly, compilation included')
    axis.set_ylabel('mean cost per step (lower is better)')
    axis.set_title('whip training')
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    output = os.path.join(os.path.dirname(__file__), 'plots', filename)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
