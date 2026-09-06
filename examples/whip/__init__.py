"""Simulation, neural network, and reinforcement learning in one Node tree.

NodeJAX is a functional, JAX-native object model meant to work the same in
any domain. This example combines three that were written without each
other: the position-based dynamics of ``examples.pbd``, the motor
controller of ``examples.actuator``, and the learners of ``examples.rl``.
A rope on a motor-driven arm is the plant; a recurrent policy commands
the motors; SHAC and PPO train it and MPPI plans on it, each from one
program, with nothing adapted at the seams.
"""

from examples.whip.plant import (
    PIVOT,
    Arm,
    HitCost,
    IdealActuator,
    Whip,
    WhipStep,
    joint_forces,
    joint_mechanical,
    link_motion,
    whip_actuator,
    whip_constraints,
    whip_particles,
    whip_rope,
)


__all__ = [
    'PIVOT',
    'Arm',
    'HitCost',
    'IdealActuator',
    'Whip',
    'WhipStep',
    'joint_forces',
    'joint_mechanical',
    'link_motion',
    'whip_actuator',
    'whip_constraints',
    'whip_particles',
    'whip_rope',
]
