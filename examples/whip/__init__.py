"""Couples PBD rope dynamics, FOC motor-actuator stacks, and RL/MPPI controllers into one Node tree.

Combines `examples.pbd`, `examples.actuator`, and `examples.rl` without adapter layers: a two-link arm driven by `ActuatorStack` joints swings a PBD particle rope toward sampled 2D targets under recurrent `SHAC`, `PPO`, and receding-horizon `MPPI` controllers.
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
